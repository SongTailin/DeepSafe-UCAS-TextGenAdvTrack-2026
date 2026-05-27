# Reproduce Commands

Run from the repository root:

```bash
cd /Users/stevezheng/Documents/Playground/TextGenAdvTrack-2026Spring
```

Generate rule-humanized intermediate output:

```bash
.venv/bin/python scripts/humanize_rules_evasion.py \
  --input-csv data/UCAS_AISAD_TEXT-val.csv \
  --output-csv outputs/humanize_rules_val_with_label.csv \
  --keep-label
```

Generate final evasion CSV:

```bash
.venv/bin/python scripts/evasion_233style.py \
  --input-csv outputs/humanize_rules_val_with_label.csv \
  --output-csv outputs/666_test_1.csv \
  --homoglyph-rate 0.45 \
  --space-rate 0.08 \
  --case-rate 0.04 \
  --punctuation-rate 0.20 \
  --seed 233
```

Compress for email:

```bash
zip -j outputs/666_test_1.zip outputs/666_test_1.csv
```

