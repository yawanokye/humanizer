from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analyzer import dashboard_report  # noqa: E402
from services.calibration import classification_metrics, predict_probability_with_model  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if int(item.get("label", -1)) not in {0, 1} or not str(item.get("text") or "").strip():
            raise ValueError(f"Invalid row {line_no} in {path}.")
        rows.append(item)
    return rows


def group_metrics(rows: list[dict], probabilities: list[float], key: str) -> dict[str, dict]:
    buckets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, prob in zip(rows, probabilities):
        buckets[str(row.get(key) or "unknown")].append((int(row["label"]), prob))
    return {
        name: classification_metrics([y for y, _ in values], [p for _, p in values])
        for name, values in buckets.items()
        if len({y for y, _ in values}) >= 2
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a v2.4 calibration model on an independent labelled JSONL corpus.")
    parser.add_argument("model", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    corpus = load_jsonl(args.corpus)
    probabilities = []
    for i, sample in enumerate(corpus, start=1):
        report = dashboard_report(str(sample["text"]), use_calibrator=False)
        pred = predict_probability_with_model(model, report["calibration_features"])
        probabilities.append(float(pred["probability"]))
        print(f"[{i}/{len(corpus)}] {sample.get('source_family', 'unknown')} -> {pred['percentage']}%")
    labels = [int(row["label"]) for row in corpus]
    result = {
        "sample_count": len(corpus),
        "overall": classification_metrics(labels, probabilities),
        "by_source_family": group_metrics(corpus, probabilities, "source_family"),
        "by_discipline": group_metrics(corpus, probabilities, "discipline"),
        "by_document_type": group_metrics(corpus, probabilities, "document_type"),
        "by_editing_level": group_metrics(corpus, probabilities, "editing_level"),
        "note": "These are held-out metrics only if the evaluation corpus was not used to train or tune the model.",
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
