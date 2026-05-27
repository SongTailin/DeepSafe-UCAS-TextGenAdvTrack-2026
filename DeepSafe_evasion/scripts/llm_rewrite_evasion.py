#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """You rewrite machine-generated text so it reads like a natural human-written answer.

Rules:
- Preserve the original language, meaning, facts, names, numbers, and order of information.
- Do not add new facts, citations, URLs, claims, or personal experiences.
- Reduce formulaic AI style: avoid overusing "In conclusion", "It is important to note", "Furthermore", "Moreover", "Firstly/Secondly/Lastly".
- Make sentence lengths less uniform. Mix short and medium sentences.
- Keep the output roughly 80%-120% of the original length.
- Preserve lists, quotes, code, equations, and non-English text when present.
- Return only the rewritten text, with no explanation.
"""


def clean_row(row):
    return {key.lstrip("\ufeff"): value for key, value in row.items()}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [clean_row(row) for row in csv.DictReader(f)]


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"prompt": row["prompt"], "text": row["text"]})


def load_cache(path):
    path = Path(path)
    if not path.exists():
        return {}
    cache = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cache[int(item["index"])] = item["text"]
    return cache


def append_cache(path, index, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"index": index, "text": text}, ensure_ascii=False) + "\n")


def chat_completion(base_url, api_key, model, prompt, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def build_prompt(prompt, text):
    return f"""Rewrite the response to sound naturally human-written while preserving the original meaning.

Instruction / prompt:
{prompt}

Original response:
{text}

Rewritten response:"""


def should_rewrite(row, assume_all_machine):
    if assume_all_machine:
        return True
    return str(row.get("label", "")) == "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cache-jsonl", default="outputs/llm_rewrite_cache.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--assume-all-machine", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set LLM_API_KEY or OPENAI_API_KEY.")

    rows = read_csv(args.input_csv)
    cache = load_cache(args.cache_jsonl)
    rewritten_count = 0

    for idx, row in enumerate(rows):
        if not should_rewrite(row, args.assume_all_machine):
            continue
        if idx in cache:
            row["text"] = cache[idx]
            continue
        if args.limit and rewritten_count >= args.limit:
            break

        prompt = build_prompt(row.get("prompt", ""), row.get("text", ""))
        for attempt in range(5):
            try:
                rewritten = chat_completion(
                    args.base_url,
                    args.api_key,
                    args.model,
                    prompt,
                    args.temperature,
                    args.timeout,
                )
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"index={idx} request failed: {exc}; retrying in {wait}s")
                time.sleep(wait)

        row["text"] = rewritten
        append_cache(args.cache_jsonl, idx, rewritten)
        rewritten_count += 1
        print(f"rewrote index={idx} count={rewritten_count}")
        time.sleep(args.sleep)

    for idx, text in cache.items():
        if 0 <= idx < len(rows) and should_rewrite(rows[idx], args.assume_all_machine):
            rows[idx]["text"] = text

    write_csv(rows, args.output_csv)
    print(f"Saved {len(rows)} rows to {args.output_csv}; new rewrites={rewritten_count}; cached={len(cache)}")


if __name__ == "__main__":
    main()
