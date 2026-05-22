#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AttackConfig:
    name: str
    p_homo: float
    p_case: float
    p_space: float


class HomoglyphAttack:
    def __init__(self) -> None:
        self.mapping = {
            "a": ["а"],
            "A": ["А", "Α"],
            "B": ["В", "Β"],
            "e": ["е"],
            "E": ["Е", "Ε"],
            "c": ["с"],
            "p": ["р"],
            "K": ["К", "Κ"],
            "O": ["О", "Ο"],
            "P": ["Р", "Ρ"],
            "M": ["М", "Μ"],
            "H": ["Н", "Η"],
            "T": ["Т", "Τ"],
            "X": ["Х", "Χ"],
            "C": ["С"],
            "y": ["у"],
            "o": ["о"],
            "x": ["х"],
            "I": ["І", "Ι"],
            "i": ["і"],
            "N": ["Ν"],
            "Z": ["Ζ"],
        }

    def attack(self, text: str, ratio: float, rng: random.Random) -> str:
        chars = list(text)
        for index, char in enumerate(chars):
            if char in self.mapping and rng.random() < ratio:
                chars[index] = rng.choice(self.mapping[char])
        return "".join(chars)


class UpperLowerFlipAttack:
    def __init__(self) -> None:
        self.pattern = re.compile(r"\w+", flags=re.UNICODE)

    def attack(self, text: str, ratio: float, rng: random.Random) -> str:
        indices = [match.start() for match in self.pattern.finditer(text) if text[match.start()].isalpha()]
        if not indices or ratio <= 0:
            return text
        count = min(len(indices), max(1, int(round(len(indices) * ratio))))
        flip_indices = rng.sample(indices, count)
        chars = list(text)
        for index in flip_indices:
            chars[index] = chars[index].lower() if chars[index].isupper() else chars[index].upper()
        return "".join(chars)


class WhiteSpaceAttack:
    def attack(self, text: str, ratio: float, rng: random.Random) -> str:
        if ratio <= 0:
            return text
        parts = text.split(" ")
        if len(parts) <= 1:
            return text
        count = int(len(parts) * ratio)
        if count <= 0:
            return text
        indices = sorted(rng.choices(range(len(parts)), k=count))
        for index in indices:
            parts[index] += " "
        return " ".join(parts)


def stable_long_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest, 16) & ((1 << 63) - 1)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = load_jsonl(path)
        normalized = []
        for row in rows:
            item = {"prompt": row["prompt"], "text": row["text"], "id": row.get("id", stable_long_hash(row["text"]))}
            if "label" in row:
                item["label"] = int(row["label"])
            normalized.append(item)
        return normalized

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            prompt = row.get("prompt") or row.get("\ufeffprompt")
            text = row.get("text")
            if prompt is None or text is None:
                raise ValueError(f"Missing prompt/text columns in {path}")
            item = {"prompt": str(prompt), "text": str(text), "id": stable_long_hash(str(text))}
            if "label" in row and row["label"] != "":
                item["label"] = int(row["label"])
            rows.append(item)
    return rows


def load_policy(path: str) -> AttackConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    configs = [AttackConfig(**cfg) for cfg in payload["configs"]]
    probs = payload["probabilities"]
    best = max(configs, key=lambda cfg: float(probs.get(cfg.name, 0.0)))
    return best


def attack_row(row: dict, config: AttackConfig, rng: random.Random, homoglyph: HomoglyphAttack, case_attack: UpperLowerFlipAttack, whitespace: WhiteSpaceAttack) -> dict:
    attacked = dict(row)
    text = attacked["text"]
    text = homoglyph.attack(text, config.p_homo, rng)
    text = case_attack.attack(text, config.p_case, rng)
    text = whitespace.attack(text, config.p_space, rng)
    attacked["text"] = text
    attacked["id"] = stable_long_hash(text)
    return attacked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--policy-json", default=str(ROOT / "attack_policy.json"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-all", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input_path)
    config = load_policy(args.policy_json)
    rng = random.Random(args.seed)
    homoglyph = HomoglyphAttack()
    case_attack = UpperLowerFlipAttack()
    whitespace = WhiteSpaceAttack()

    attacked_rows = []
    attacked_count = 0
    for row in rows:
        item = dict(row)
        should_attack = args.attack_all or ("label" in item and int(item["label"]) == 0)
        if should_attack:
            item = attack_row(item, config, rng, homoglyph, case_attack, whitespace)
            attacked_count += 1
        attacked_rows.append(item)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["prompt", "text"])
        writer.writeheader()
        for row in attacked_rows:
            writer.writerow({"prompt": row["prompt"], "text": row["text"]})

    print(f"Saved evasion csv to {output_path}")
    print(f"Applied policy {config.name} to {attacked_count} rows")


if __name__ == "__main__":
    main()
