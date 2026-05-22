#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent


class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, adapter_path: str | None, output_hidden_states: bool, use_pooling: str = "max") -> None:
        super().__init__()
        if output_hidden_states:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.use_pooling = use_pooling

    def pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor, hidden_states: bool) -> torch.Tensor:
        if hidden_states:
            if self.use_pooling == "max":
                masked = model_output.masked_fill(~attention_mask[None, ..., None].bool(), float("-inf"))
                emb, _ = masked.max(dim=2)
            elif self.use_pooling == "average":
                masked = model_output.masked_fill(~attention_mask[None, ..., None].bool(), 0.0)
                emb = masked.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:, :, 0]
            emb = emb.permute(1, 0, 2)
        else:
            if self.use_pooling == "max":
                masked = model_output.masked_fill(~attention_mask[..., None].bool(), float("-inf"))
                emb, _ = masked.max(dim=1)
            elif self.use_pooling == "average":
                masked = model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
                emb = masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:, 0]
        return emb

    def forward(self, encoded_batch: dict[str, torch.Tensor], hidden_states: bool = False) -> torch.Tensor:
        model_output = self.model(**encoded_batch)
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            if hidden_states:
                model_output = torch.stack(model_output["hidden_states"], dim=0)
            else:
                model_output = model_output["last_hidden_state"]
        return self.pooling(model_output, encoded_batch["attention_mask"], hidden_states)


class FrozenEncoder:
    def __init__(self, model_name: str, adapter_path: str | None, device: str, pooling: str) -> None:
        model = TextEmbeddingModel(
            model_name=model_name,
            adapter_path=adapter_path,
            output_hidden_states=True,
            use_pooling=pooling,
        )
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model
        self.device = device

    def encode_texts(self, texts: list[str], batch_size: int, max_length: int, layer: int) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.model.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                embeddings = self.model(encoded, hidden_states=True)
                chunks.append(F.normalize(embeddings[:, layer, :], dim=-1).cpu())
        return torch.cat(chunks, dim=0)


class LinearDetectorHead(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x)).squeeze(-1)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def rows_to_texts(rows: list[dict]) -> list[str]:
    return [str(row["text"]) for row in rows]


def predict_detector_head(head: nn.Module, embeddings: torch.Tensor, batch_size: int, device: str) -> np.ndarray:
    head.to(device)
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            outputs.append(torch.sigmoid(head(batch)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def score_knn_probs(query_embeddings: torch.Tensor, bank_embeddings: torch.Tensor, bank_labels: np.ndarray, k: int) -> np.ndarray:
    sims = query_embeddings.float() @ bank_embeddings.float().T
    top_scores, top_indices = torch.topk(sims, k=min(k, bank_embeddings.size(0)), dim=1, largest=True, sorted=True)
    probs = []
    for row_scores, row_indices in zip(top_scores.numpy(), top_indices.numpy()):
        votes = np.zeros(2, dtype=np.float32)
        for score, index in zip(row_scores, row_indices):
            votes[bank_labels[index]] += float(score)
        shifted = np.exp(votes - votes.max())
        probs.append(float((shifted / shifted.sum())[1]))
    return np.array(probs, dtype=np.float32)


def build_combined_scores(cls_probs: np.ndarray, knn_probs: np.ndarray, cls_weight: float, knn_weight: float) -> np.ndarray:
    total = cls_weight + knn_weight
    return (cls_weight * cls_probs + knn_weight * knn_probs) / total


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


def load_dataset(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = load_jsonl(path)
        return [{"prompt": row["prompt"], "text": row["text"], "id": row.get("id", stable_long_hash(row["text"]))} for row in rows]

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


def export_official_xlsx(output_path: Path, rows: list[dict], probs: np.ndarray, elapsed_seconds: float) -> None:
    def col_name(index: int) -> str:
        name = ""
        while index > 0:
            index, rem = divmod(index - 1, 26)
            name = chr(65 + rem) + name
        return name

    def cell_xml(ref: str, value: object) -> str:
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    def worksheet_xml(rows_2d: list[list[object]]) -> str:
        xml_rows = []
        for row_index, row_values in enumerate(rows_2d, start=1):
            cells = []
            for col_index, value in enumerate(row_values, start=1):
                cells.append(cell_xml(f"{col_name(col_index)}{row_index}", value))
            xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(xml_rows)}</sheetData>'
            "</worksheet>"
        )

    predictions_rows = [["prompt", "text_prediction"]]
    predictions_rows.extend([[row["prompt"], float(prob)] for row, prob in zip(rows, probs)])
    time_rows = [["Data Volume", "Time"], [len(rows), float(elapsed_seconds)]]

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="predictions" sheetId="1" r:id="rId1"/>
    <sheet name="time" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"""

    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>"""

    core_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>official_predictions</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml(predictions_rows))
        zf.writestr("xl/worksheets/sheet2.xml", worksheet_xml(time_rows))
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Path or HF id for the public base RoBERTa-large")
    parser.add_argument("--adapter-path", default=str(ROOT), help="Path to the released LoRA adapter directory")
    parser.add_argument("--detector-head", default=str(ROOT / "detector_head.pt"))
    parser.add_argument("--retrieval-bank", default=str(ROOT / "retrieval_bank.pt"))
    parser.add_argument("--input-path", required=True, help="CSV or JSONL input file")
    parser.add_argument("--output-xlsx", required=True, help="Official-format xlsx output path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--eval-cls-weight", type=float, default=0.4)
    parser.add_argument("--eval-knn-weight", type=float, default=0.6)
    parser.add_argument("--knn-k", type=int, default=2)
    parser.add_argument("--pooling", default="max")
    args = parser.parse_args()

    device = choose_device(args.device)
    rows = load_dataset(args.input_path)
    print(f"Loaded {len(rows)} rows from {args.input_path}")

    adapter_path = args.adapter_path if args.adapter_path and Path(args.adapter_path).exists() else None
    encoder = FrozenEncoder(
        model_name=args.base_model,
        adapter_path=adapter_path,
        device=device,
        pooling=args.pooling,
    )

    head_data = torch.load(args.detector_head, map_location="cpu", weights_only=False)
    head = LinearDetectorHead(dim=int(head_data["dim"]), dropout=float(head_data["dropout"]))
    head.load_state_dict(head_data["state_dict"])
    head.to(device)
    head.eval()

    bank_data = torch.load(args.retrieval_bank, map_location="cpu", weights_only=False)
    bank_embeddings = bank_data["bank_embeddings"].float()
    bank_labels = np.array(bank_data["bank_labels"], dtype=np.int64)

    start = time.time()
    embeddings = encoder.encode_texts(rows_to_texts(rows), args.encode_batch_size, args.max_length, args.layer)
    cls_probs = predict_detector_head(head, embeddings, args.eval_batch_size, device)
    knn_probs = score_knn_probs(embeddings, bank_embeddings, bank_labels, args.knn_k)
    probs = build_combined_scores(cls_probs, knn_probs, args.eval_cls_weight, args.eval_knn_weight)
    elapsed = time.time() - start

    output_path = Path(args.output_xlsx)
    export_official_xlsx(output_path, rows, probs, elapsed)
    print(f"Saved official predictions to {output_path}")
    print(f"Processed {len(rows)} rows in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
