# DeepSafe Official Repository

This is the official repository of **DeepSafe** for the **generative text and adversarial task** in the 2026 UCAS course **Artificial Intelligence Security and Adversarial Defense**.

This repository provides the scripts and released artifacts required to evaluate our methods on the official task format.

## 1. Download The Released Model Files

After cloning this repository, make sure that the released DeepSafe artifact files are available directly in the repository root:

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
- the script expects the released detector files in the current repository root by default

### Included Test-1 Label Result                                                                  
                                                                                                    
The file                                                                                          
                                                                                                    
- `UCAS_AISAD_TEXT-test1_label.csv`
                                                                                                 
is our submitted label result on the official `test_1` set.

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
- if labels are absent, the script cannot infer which rows are machine-generated from the file alone, so strict selective modification would require additional external row-level information
- output follows the official CSV format
- the script expects the released attacker policy file in the current repository root by default

## 4. Technical Overview

Our method is organized into three stages and is built around a simple principle: the detector and the attacker should not be designed in isolation, because each one defines the operating environment of the other.

### Stage 1: Pretrained Detection Backbone

We begin from a pretrained encoder-based text detector backbone built on top of a large pretrained language model. This stage provides the semantic representation that later supports both classification-style and retrieval-style judgment.

The motivation for using an encoder-style backbone is that it offers a stable sentence-level representation space, which is especially useful when:

- the detector needs to compare human and machine texts at the representation level
- the system must support both direct scoring and nearest-neighbor style evidence
- the model should remain robust under moderate text perturbation

This stage therefore establishes the semantic foundation of the whole framework rather than directly defining the final decision rule.

### Stage 2: Adversarial Co-Training

We then perform adversarial co-training between:

- a detector branch that combines semantic classification and retrieval-style evidence
- an attacker branch that searches for effective text perturbations

This stage is used to discover a strong detector and a strong base attacker under the official task setting.

The key idea is not to optimize the detector on clean data alone. Instead:

- the attacker continuously searches for stronger perturbations
- the detector is repeatedly exposed to harder adversarial machine texts
- the resulting training loop gradually sharpens both robustness and attack efficiency

This creates a detector that is less dependent on clean-text regularities and an attacker that is optimized against the detector actually used in the system.

### Stage 3: Attacker-Focused Refinement

After the detector becomes stable, we continue refining the attacker separately against a fixed detector. This stage is used to obtain a stronger final evasion policy with improved attack effectiveness and reduced unnecessary perturbation.

This separation is important because the best detector and the best attacker do not necessarily appear at the same training point. Once the detector is sufficiently stable, freezing it makes later attacker improvement easier to interpret and prevents the target from moving continuously during optimization.

## 5. Main Ideas And Improvements

Compared with a plain baseline, our released solution introduces the following improvements:

- a detector that combines semantic scoring and retrieval-style evidence instead of relying on a single signal
- a multi-stage adversarial optimization process rather than a one-shot attack design
- a refined attacker that prefers lower-cost perturbations while maintaining attack effectiveness
- a final release that supports direct evaluation on the official task format without reimplementing internal utilities

### 5.1 Detector As A Dual-Branch Decision System

Our detector does not rely on a single decision signal. Instead, it combines:

1. **Semantic scoring**
   - a lightweight detector head operates on top of backbone text embeddings
   - this branch learns a global decision boundary between human-written and machine-generated text

2. **Retrieval-style scoring**
   - a retrieval bank stores reference embeddings
   - for each input, the detector retrieves similar examples and uses neighborhood structure as additional evidence

The rationale is that the two branches capture different types of information:

- the semantic branch captures global discriminative structure
- the retrieval branch captures local similarity structure

In practice, this fusion improves robustness because adversarial perturbations that are sufficient to move a single classifier score are not always sufficient to simultaneously change the local neighborhood evidence in a meaningful way.

### 5.2 Attacker As A Strategy Distribution

The attacker is not implemented as one fixed perturbation script. Instead, it is represented as a distribution over perturbation strategies. These strategies include different mixtures of:

- homoglyph substitution
- case perturbation
- whitespace perturbation

For each machine text, the attacker generates multiple candidates using different perturbation configurations, and the detector then identifies which candidate is hardest to classify correctly. The attack process therefore becomes adaptive rather than static.

This design allows the system to discover that:

- some perturbations are too weak
- some perturbations are too destructive
- the strongest perturbations are often the ones that best balance effectiveness and perturbation cost

### 5.3 Multi-Stage Optimization Instead Of One-Shot Training

A one-shot pipeline usually produces one of two weak outcomes:

- a detector that performs well on clean text but fails under attack
- an attacker that is manually designed but not truly strong against the trained detector

Our multi-stage approach addresses this by first establishing a strong representation backbone, then jointly improving detector and attacker, and finally refining the attacker against a fixed defense. This staged procedure yields a more stable final system.

## 6. Innovation Points

The main design innovations of this release are:

- combining representation-based detection with retrieval-based judgment
- using adversarial interaction to jointly improve robustness and attack strength
- separating late-stage attacker refinement from the earlier detector-attacker co-training stage
- emphasizing efficient perturbations instead of only stronger perturbations

More concretely, the released method can be understood through three innovation points:

### 6.1 Representation + Retrieval Fusion

Instead of reducing the task to plain binary classification, the detector integrates:

- semantic representation learning
- retrieval-based local evidence

This gives the detector a stronger structural prior and improves robustness under perturbation.

### 6.2 Adversarially Guided Detector Improvement

The detector is improved not only by seeing clean samples, but by repeatedly confronting stronger attacker-generated texts. This allows the detector to move beyond clean-data pattern matching and become more robust to evasion attempts.

### 6.3 Efficient-Attack Preference

The attacker is not optimized merely to maximize visible distortion. It is optimized to maximize evasion effect while keeping perturbation cost relatively low. This is especially important in realistic adversarial settings, where strong attacks must still preserve the overall readability and content of the text.

## 7. Repository Structure

- `adapter_config.json`, `adapter_model.safetensors`
  - released detector adapter
- `detector_head.pt`, `retrieval_bank.pt`
  - released detector head and retrieval bank
- `UCAS_AISAD_TEXT-test1_label.csv`
   - our submitted label result on the official `test_1` set
- `attack_policy.json`
  - released attacker policy
- `predict_detection.py`
  - direct official-format detection inference
- `generate_evasion.py`
  - direct official-format evasion generation
- `requirements.txt`
  - minimal inference dependencies
