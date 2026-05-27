#!/usr/bin/env python3
import argparse
import csv
import random
import re
from pathlib import Path


PHRASE_REPLACEMENTS = [
    (r"\bIt is important to note that\b", ["One thing worth noting is that", "A useful detail here is that"]),
    (r"\bIt should be noted that\b", ["Worth noting:", "A useful point is that"]),
    (r"\bFurthermore\b", ["Also", "On top of that", "Another point is"]),
    (r"\bMoreover\b", ["Also", "Besides that", "Another thing is"]),
    (r"\bIn conclusion\b", ["Overall", "In short", "Taken together"]),
    (r"\bIn summary\b", ["Overall", "Put simply", "In short"]),
    (r"\bFirstly\b", ["First", "To start with"]),
    (r"\bSecondly\b", ["Second", "Another point"]),
    (r"\bLastly\b", ["Finally", "One last point"]),
    (r"\bTherefore\b", ["So", "For that reason"]),
    (r"\bHowever\b", ["But", "That said"]),
    (r"\bThis is because\b", ["That is because", "The reason is that"]),
    (r"\bThere are several reasons why\b", ["A few reasons explain why", "There are a few reasons why"]),
    (r"\bplays a crucial role\b", ["matters a lot", "is a big part of this"]),
    (r"\bis a complex and multifaceted\b", ["is not a simple", "has several sides as a"]),
    (r"\bvarious factors\b", ["several factors", "a few things"]),
    (r"\ba wide range of\b", ["many", "a range of"]),
    (r"\butilize\b", ["use"]),
    (r"\bfacilitate\b", ["help"]),
    (r"\bdemonstrate\b", ["show"]),
    (r"\bapproximately\b", ["about"]),
    (r"\bnumerous\b", ["many"]),
]

CONTRACTIONS = [
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwill not\b", "won't"),
    (r"\bcannot\b", "can't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
]


def clean_row(row):
    return {key.lstrip("\ufeff"): value for key, value in row.items()}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [clean_row(row) for row in csv.DictReader(f)]


def write_csv(rows, path, keep_label=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt", "text", "label"] if keep_label else ["prompt", "text"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"prompt": row["prompt"], "text": row["text"]}
            if keep_label:
                out["label"] = row.get("label", "")
            writer.writerow(out)


def mostly_latin(text):
    latin = sum("a" <= ch.lower() <= "z" for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    cyr = sum("\u0400" <= ch <= "\u04ff" for ch in text)
    return latin >= max(cjk, cyr) * 2 and latin > 50


def phrase_replace(text, rng, rate):
    for pattern, choices in PHRASE_REPLACEMENTS:
        if re.search(pattern, text, flags=re.I) and rng.random() < rate:
            text = re.sub(pattern, rng.choice(choices), text, count=1, flags=re.I)
    return text


def contractions(text, rng, rate):
    for pattern, repl in CONTRACTIONS:
        if rng.random() < rate:
            text = re.sub(pattern, repl, text, flags=re.I)
    return text


def soften_numbered_transitions(text, rng):
    replacements = {
        "First,": ["First,", "For one thing,", "To start,"],
        "Second,": ["Second,", "Another piece is", "Next,"],
        "Third,": ["Third,", "There is also", "Another factor is"],
        "Finally,": ["Finally,", "One last thing:", "At the end of the day,"],
    }
    for old, choices in replacements.items():
        if old in text and rng.random() < 0.5:
            text = text.replace(old, rng.choice(choices), 1)
    return text


def split_some_long_sentences(text, rng, rate):
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for sentence in parts:
        words = sentence.split()
        if len(words) > 34 and rng.random() < rate:
            splitters = [", which ", ", because ", ", as ", ", while ", ", and "]
            lowered = sentence.lower()
            positions = []
            for splitter in splitters:
                idx = lowered.find(splitter)
                if idx > 40:
                    positions.append(idx)
            if positions:
                idx = min(positions)
                first = sentence[:idx].rstrip(", ")
                second = sentence[idx:].lstrip(", ")
                if second:
                    second = second[0].upper() + second[1:]
                out.append(first + ". " + second)
                continue
        out.append(sentence)
    return " ".join(out)


def paragraph_tweak(text, rng):
    if "\n" in text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) >= 5 and rng.random() < 0.35:
        cut = rng.randint(2, min(4, len(sentences) - 2))
        return " ".join(sentences[:cut]) + "\n\n" + " ".join(sentences[cut:])
    return text


def humanize_english(text, rng):
    text = phrase_replace(text, rng, rate=0.75)
    text = contractions(text, rng, rate=0.45)
    text = soften_numbered_transitions(text, rng)
    text = split_some_long_sentences(text, rng, rate=0.45)
    text = paragraph_tweak(text, rng)
    text = re.sub(r" +", " ", text)
    return text


def humanize_non_latin(text, rng):
    # Conservative changes only: preserve language and avoid damaging meaning.
    if rng.random() < 0.25:
        text = text.replace("。", "。\n\n", 1)
    if rng.random() < 0.20:
        text = text.replace(". ", ".\n\n", 1)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--assume-all-machine", action="store_true")
    parser.add_argument("--keep-label", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_csv(args.input_csv)
    changed = 0
    for row in rows:
        label = str(row.get("label", ""))
        if not args.assume_all_machine and label != "0":
            continue
        old = row["text"]
        if mostly_latin(old):
            row["text"] = humanize_english(old, rng)
        else:
            row["text"] = humanize_non_latin(old, rng)
        if row["text"] != old:
            changed += 1

    write_csv(rows, args.output_csv, keep_label=args.keep_label)
    print(f"Saved {len(rows)} rows to {args.output_csv}; changed={changed}")


if __name__ == "__main__":
    main()
