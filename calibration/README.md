# v2.2 calibration corpus

The detector only calls itself **calibrated** when a labelled model artifact exists. Do not train on a tiny convenience sample.

Use JSONL with one object per document or passage:

```json
{"label":0,"source_family":"human-published","discipline":"public administration","text":"..."}
{"label":1,"source_family":"gpt","discipline":"public administration","text":"..."}
```

`label=0` means known human writing and `label=1` means known AI-generated writing. Keep provenance outside the text. Include human academic work, student work, multiple model families, AI-human edited mixtures and humanized AI. Split evaluation by source family and discipline before treating the model as production calibrated.

Train with:

```bash
python scripts/train_calibration.py calibration/benchmark.jsonl
```

The trainer refuses fewer than 40 samples or fewer than 15 samples in either class. Its saved metrics are **training-set diagnostics**, not deployment claims. Validate on a held-out corpus.

Evaluate on a separate held-out corpus with:

```bash
python scripts/evaluate_calibration.py calibration/meta_classifier.json calibration/heldout.jsonl --output calibration/heldout_metrics.json
```

The evaluator reports overall metrics plus source-family and discipline metrics when each group contains both classes. Do not call training-set metrics validation performance.

## v2.3 Benchmark Lab schema

A benchmark sample can include these fields in addition to `text` and binary `label`: `provenance`, `source_family`, `discipline`, `document_type`, `editing_level`, `word_count`, and optional `external_scores`. Recommended provenance values are `human_original`, `human_edited`, `gpt_generated`, `claude_generated`, `gemini_generated`, `other_ai_generated`, `ai_human_edited`, `human_ai_edited`, `engine1_output`, `engine2_output`, `engine3_output`, and `other_humanizer_output`.

The web Benchmark Lab writes `benchmark_lab.jsonl`. The trainer compares logistic regression, Gaussian naive Bayes, and nearest-centroid models on a stratified held-out split before selecting a family. `validation_report.json` records the Validation Centre summary. External detector percentages are stored only to measure disagreement and are not training labels.
