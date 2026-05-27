#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

import predict_detection as core


def stable_long_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest, 16) & ((1 << 63) - 1)


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            prompt = row.get("prompt") or row.get("\ufeffprompt")
            text = row.get("text")
            if prompt is None or text is None:
                raise ValueError(f"Missing prompt/text columns in {path}")
            rows.append({"prompt": str(prompt), "text": str(text), "id": stable_long_hash(str(text))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--your-team-name", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="deepsafe_detector")
    parser.add_argument("--result-path", type=str, required=True)

    parser.add_argument("--base-model", type=str, default=str(Path(__file__).resolve().parent.parent / "models" / "roberta-large"))
    parser.add_argument("--adapter-path", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--detector-head", type=str, default=str(Path(__file__).resolve().parent / "detector_head.pt"))
    parser.add_argument("--retrieval-bank", type=str, default=str(Path(__file__).resolve().parent / "retrieval_bank.pt"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--eval-cls-weight", type=float, default=0.4)
    parser.add_argument("--eval-knn-weight", type=float, default=0.6)
    parser.add_argument("--knn-k", type=int, default=2)
    parser.add_argument("--pooling", type=str, default="max")
    args = parser.parse_args()

    device = core.choose_device(args.device)
    rows = load_dataset(args.data_path)
    print(f"Loaded {len(rows)} rows from {args.data_path}")
    print(f"Using model type: {args.model_type}")

    encoder = core.FrozenEncoder(
        model_name=args.base_model,
        adapter_path=args.adapter_path,
        device=device,
        pooling=args.pooling,
    )

    head_data = torch.load(args.detector_head, map_location="cpu", weights_only=False)
    head = core.LinearDetectorHead(dim=int(head_data["dim"]), dropout=float(head_data["dropout"]))
    head.load_state_dict(head_data["state_dict"])
    head.to(device)
    head.eval()

    bank_data = torch.load(args.retrieval_bank, map_location="cpu", weights_only=False)
    bank_embeddings = bank_data["bank_embeddings"].float()
    bank_labels = np.array(bank_data["bank_labels"], dtype=np.int64)

    start = time.time()
    embeddings = encoder.encode_texts(core.rows_to_texts(rows), args.encode_batch_size, args.max_length, args.layer)
    cls_probs = core.predict_detector_head(head, embeddings, args.eval_batch_size, device)
    knn_probs = core.score_knn_probs(embeddings, bank_embeddings, bank_labels, args.knn_k)
    probs = core.build_combined_scores(cls_probs, knn_probs, args.eval_cls_weight, args.eval_knn_weight)
    elapsed = time.time() - start

    result_dir = Path(args.result_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"{args.your_team_name}.xlsx"
    core.export_official_xlsx(output_path, rows, probs, elapsed)
    print(f"Saved official-format detection output to {output_path}")


if __name__ == "__main__":
    main()
