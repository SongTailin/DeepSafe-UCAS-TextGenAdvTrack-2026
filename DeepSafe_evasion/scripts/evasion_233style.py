#!/usr/bin/env python3
import argparse
import csv
import random
import re
from pathlib import Path


HOMOGLYPHS = {
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


def clean_row(row):
    return {key.lstrip("\ufeff"): value for key, value in row.items()}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [clean_row(row) for row in csv.DictReader(f)]


def write_evasion_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"prompt": row["prompt"], "text": row["text"]})


def homoglyph(text, rate):
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in HOMOGLYPHS and random.random() < rate:
            chars[i] = random.choice(HOMOGLYPHS[ch])
    return "".join(chars)


def flip_word_initial_case(text, rate):
    chars = list(text)
    starts = [m.start() for m in re.finditer(r"\b[A-Za-z]", text)]
    for idx in starts:
        if random.random() < rate:
            chars[idx] = chars[idx].lower() if chars[idx].isupper() else chars[idx].upper()
    return "".join(chars)


def whitespace(text, rate):
    parts = text.split(" ")
    if len(parts) <= 2:
        return text
    for i in range(len(parts)):
        if parts[i] and random.random() < rate:
            parts[i] += " "
    return " ".join(parts)


def punctuation_light(text, rate):
    replacements = {
        ", ": [", ", "; ", " - "],
        ". ": [". ", "\n\n", " "],
        "Furthermore, ": ["Also, ", "More importantly, "],
        "In conclusion, ": ["Overall, ", "In short, "],
        "It is important to note that ": ["One thing worth noting is that "],
    }
    for old, candidates in replacements.items():
        if old in text and random.random() < rate:
            text = text.replace(old, random.choice(candidates), 1)
    return text


def attack_text(text, args):
    if args.homoglyph_rate:
        text = homoglyph(text, args.homoglyph_rate)
    if args.case_rate:
        text = flip_word_initial_case(text, args.case_rate)
    if args.space_rate:
        text = whitespace(text, args.space_rate)
    if args.punctuation_rate:
        text = punctuation_light(text, args.punctuation_rate)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--homoglyph-rate", type=float, default=0.35)
    parser.add_argument("--case-rate", type=float, default=0.08)
    parser.add_argument("--space-rate", type=float, default=0.08)
    parser.add_argument("--punctuation-rate", type=float, default=0.25)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = read_csv(args.input_csv)
    changed = 0
    for row in rows:
        label = str(row.get("label", ""))
        if label == "0":
            row["text"] = attack_text(row["text"], args)
            changed += 1
    write_evasion_csv(rows, args.output_csv)
    print(f"Saved {len(rows)} rows to {args.output_csv}; modified {changed} machine-text rows")


if __name__ == "__main__":
    main()
