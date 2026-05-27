#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path


def clean_columns(row):
    return {key.lstrip("\ufeff"): value for key, value in row.items()}


def read_dataset(path, label_path=None):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = clean_columns(row)
            rows.append(
                {
                    "prompt": row.get("prompt", ""),
                    "text": row.get("text", ""),
                    "label": row.get("label", ""),
                }
            )

    if label_path:
        labels = []
        with Path(label_path).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row = clean_columns(row)
                labels.append(row["label"])
        if len(labels) != len(rows):
            raise ValueError(f"Label count {len(labels)} != row count {len(rows)}")
        for row, label in zip(rows, labels):
            row["label"] = label

    return rows


def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ngrams(text, min_n=3, max_n=5):
    text = normalize(text)
    if not text:
        return
    padded = f" {text} "
    for n in range(min_n, max_n + 1):
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            yield padded[i : i + n]


def build_vocab(rows, min_n, max_n, vocab_size):
    df = Counter()
    for row in rows:
        df.update(set(ngrams(row["text"], min_n, max_n)))
    return {token: idx for idx, (token, _) in enumerate(df.most_common(vocab_size))}


def train_nb(rows, min_n=3, max_n=5, vocab_size=200000, alpha=0.5):
    vocab = build_vocab(rows, min_n, max_n, vocab_size)
    class_doc_counts = Counter()
    class_token_counts = {"0": Counter(), "1": Counter()}
    class_total_tokens = Counter()

    for row in rows:
        label = str(row["label"])
        if label not in {"0", "1"}:
            continue
        class_doc_counts[label] += 1
        counts = Counter(token for token in ngrams(row["text"], min_n, max_n) if token in vocab)
        class_token_counts[label].update(counts)
        class_total_tokens[label] += sum(counts.values())

    total_docs = class_doc_counts["0"] + class_doc_counts["1"]
    if total_docs == 0:
        raise ValueError("No labeled training rows found")

    vocab_n = max(len(vocab), 1)
    log_prior = {
        label: math.log((class_doc_counts[label] + 1.0) / (total_docs + 2.0))
        for label in ("0", "1")
    }
    denominators = {
        label: class_total_tokens[label] + alpha * vocab_n for label in ("0", "1")
    }
    default_log_prob = {
        label: math.log(alpha / denominators[label]) for label in ("0", "1")
    }
    log_likelihood = {"0": {}, "1": {}}
    for token in vocab:
        for label in ("0", "1"):
            count = class_token_counts[label][token]
            log_likelihood[label][token] = math.log((count + alpha) / denominators[label])

    return {
        "min_n": min_n,
        "max_n": max_n,
        "vocab": vocab,
        "log_prior": log_prior,
        "default_log_prob": default_log_prob,
        "log_likelihood": log_likelihood,
    }


def predict_one(model, text):
    scores = {
        "0": model["log_prior"]["0"],
        "1": model["log_prior"]["1"],
    }
    counts = Counter(
        token
        for token in ngrams(text, model["min_n"], model["max_n"])
        if token in model["vocab"]
    )
    for token, count in counts.items():
        scores["0"] += count * model["log_likelihood"]["0"].get(
            token, model["default_log_prob"]["0"]
        )
        scores["1"] += count * model["log_likelihood"]["1"].get(
            token, model["default_log_prob"]["1"]
        )
    log_odds = max(min(scores["1"] - scores["0"], 50), -50)
    return 1.0 / (1.0 + math.exp(-log_odds))


def predict(model, rows):
    return [predict_one(model, row["text"]) for row in rows]


def auc_score(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(1 for label in labels if label == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in pairs[i:j] if label == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def metrics(rows, scores, threshold=0.5):
    labels = [int(row["label"]) for row in rows]
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auc": auc_score(labels, scores),
        "accuracy": (tp + tn) / len(labels),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_submission(rows, scores, elapsed, output_path):
    from openpyxl import Workbook

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "predictions"
    ws.append(["prompt", "text_prediction"])
    for row, score in zip(rows, scores):
        ws.append([row["prompt"], float(score)])

    time_ws = wb.create_sheet("time")
    time_ws.append(["Data Volume", "Time"])
    time_ws.append([len(rows), float(elapsed)])
    wb.save(output_path)


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = dict(model)
    compact["vocab"] = list(model["vocab"].keys())
    with path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)


def load_model(path):
    with Path(path).open(encoding="utf-8") as f:
        model = json.load(f)
    model["vocab"] = {token: idx for idx, token in enumerate(model["vocab"])}
    model["min_n"] = int(model["min_n"])
    model["max_n"] = int(model["max_n"])
    return model


def split_rows(rows, valid_ratio, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_valid = int(len(shuffled) * valid_ratio)
    return shuffled[n_valid:], shuffled[:n_valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-label-csv")
    parser.add_argument("--predict-csv")
    parser.add_argument("--team-name", default="baseline")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-out", default="models/char_nb.json")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=120000)
    parser.add_argument("--fit-all-for-predict", action="store_true")
    parser.add_argument("--load-model")
    args = parser.parse_args()

    rows = read_dataset(args.train_csv, args.train_label_csv)
    labeled = [row for row in rows if str(row["label"]) in {"0", "1"}]
    if args.load_model:
        model = load_model(args.load_model)
    else:
        train_rows, valid_rows = split_rows(labeled, args.valid_ratio, args.seed)
        start = time.perf_counter()
        model = train_nb(train_rows, args.min_n, args.max_n, args.vocab_size)
        valid_scores = predict(model, valid_rows)
        elapsed = time.perf_counter() - start
        result = metrics(valid_rows, valid_scores)
        print(
            "Validation "
            f"AUC={result['auc']:.4f} "
            f"ACC={result['accuracy']:.4f} "
            f"F1={result['f1']:.4f} "
            f"P={result['precision']:.4f} "
            f"R={result['recall']:.4f} "
            f"time={elapsed:.2f}s"
        )
        save_model(model, args.model_out)

    if args.predict_csv:
        if args.fit_all_for_predict:
            model = train_nb(labeled, args.min_n, args.max_n, args.vocab_size)
            save_model(model, args.model_out)
        predict_rows = read_dataset(args.predict_csv)
        pred_start = time.perf_counter()
        scores = predict(model, predict_rows)
        pred_elapsed = time.perf_counter() - pred_start
        output_path = Path(args.output_dir) / f"{args.team_name}.xlsx"
        write_submission(predict_rows, scores, pred_elapsed, output_path)
        print(f"Saved submission to {output_path}")
        print(f"Predicted {len(predict_rows)} rows in {pred_elapsed:.2f}s")


if __name__ == "__main__":
    main()
