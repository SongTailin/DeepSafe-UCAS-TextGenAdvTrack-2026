# DeepSafe Official Repository

This is the official repository of **DeepSafe** for the **generative text and adversarial task** in the 2026 UCAS course **Artificial Intelligence Security and Adversarial Defense**.

This repository provides the scripts and released artifacts required to evaluate our methods on the official task format.

## 1. Download The Released Model Files

After cloning this repository, download the released model files directly from this repository and place them in the repository root:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.json`
- `merges.txt`
- `detector_head.pt`
- `retrieval_bank.pt`
- `attack_policy.json`

In addition, the detection script requires the public base encoder:

- `RoBERTa-large`

You may either:

1. download the base model manually and point the script to its local path, or
2. place it in your own model directory and use the corresponding command-line arguments.

## 2. Detection Evaluation On The Official Task

The official detection task expects:

- input: CSV or JSONL containing `prompt,text`
- output: an `.xlsx` file with:
  - sheet `predictions`: `prompt,text_prediction`
  - sheet `time`: `Data Volume,Time`

Run detection inference as follows:

```bash
python predict_detection.py \
  --base-model /path/to/roberta-large \
  --adapter-path . \
  --input-path /path/to/UCAS_AISAD_TEXT-test_1.csv \
  --output-xlsx ./your_team_name.xlsx \
  --device cuda:0
```

Notes:

- higher `text_prediction` means more likely human-written
- the script directly exports the official-format `.xlsx`
- input may also be `.jsonl`

## 3. Evasion Evaluation On The Official Task

The official evasion task expects:

- input: CSV or JSONL containing `prompt,text,label(optional)`
- output: UTF-8 CSV containing `prompt,text`

Run evasion generation as follows:

```bash
python generate_evasion.py \
  --input-path /path/to/UCAS_AISAD_TEXT-val.csv \
  --output-csv ./your_team_name_test.csv
```

Notes:

- if labels are present, only `label=0` rows are modified
- if labels are absent, the script cannot automatically distinguish machine rows from human rows
- output follows the official CSV format

## 4. Technical Overview

Our method is organized into three stages.

### Stage 1: Pretrained Detection Backbone

We begin from a pretrained encoder-based text detector backbone built on top of a large pretrained language model. This stage provides the base semantic representation used by the downstream detector.

### Stage 2: Adversarial Co-Training

We then perform adversarial co-training between:

- a detector branch that combines semantic classification and retrieval-style evidence
- an attacker branch that searches for effective text perturbations

This stage is used to discover a strong detector and a strong base attacker under the official task setting.

### Stage 3: Attacker-Focused Refinement

After the detector becomes stable, we continue refining the attacker separately against a fixed detector. This stage is used to obtain a stronger final evasion policy with improved attack effectiveness and reduced unnecessary perturbation.

## 5. Main Ideas And Improvements

Compared with a plain baseline, our released solution introduces the following improvements:

- a detector that combines semantic scoring and retrieval-style evidence instead of relying on a single signal
- a multi-stage adversarial optimization process rather than a one-shot attack design
- a refined attacker that prefers lower-cost perturbations while maintaining attack effectiveness
- a final release that supports direct evaluation on the official task format without reimplementing internal utilities

## 6. Innovation Points

The main design innovations of this release are:

- combining representation-based detection with retrieval-based judgment
- using adversarial interaction to jointly improve robustness and attack strength
- separating late-stage attacker refinement from the earlier detector-attacker co-training stage
- emphasizing efficient perturbations instead of only stronger perturbations

## 7. Repository Structure

- `adapter_config.json`, `adapter_model.safetensors`
  - released detector adapter
- `detector_head.pt`, `retrieval_bank.pt`
  - released detector head and retrieval bank
- `attack_policy.json`
  - released attacker policy
- `predict_detection.py`
  - direct official-format detection inference
- `generate_evasion.py`
  - direct official-format evasion generation
- `requirements.txt`
  - minimal inference dependencies
