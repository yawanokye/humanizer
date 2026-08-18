from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analyzer import dashboard_report  # noqa: E402
from services.calibration import save_model, train_best_meta_classifier  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if int(item.get("label", -1)) not in {0, 1} or not str(item.get("text") or "").strip():
            raise ValueError(f"Invalid row {line_no}: each sample needs label 0/1 and non-empty text.")
        rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Scholarly Humanizer v2.3 held-out-selected calibration meta-classifier.")
    parser.add_argument("corpus", type=Path, help="JSONL corpus with label, source_family, discipline and text fields")
    parser.add_argument("--output", type=Path, default=ROOT / "calibration" / "meta_classifier.json")
    args = parser.parse_args()
    corpus = load_jsonl(args.corpus)
    feature_rows = []
    for i, sample in enumerate(corpus, start=1):
        report = dashboard_report(str(sample["text"]), use_calibrator=False)
        feature_rows.append({"label": int(sample["label"]), "features": report["calibration_features"]})
        print(f"[{i}/{len(corpus)}] extracted {sample.get('source_family', 'unknown')}")
    model = train_best_meta_classifier(feature_rows)
    model["trained_at"] = datetime.now(UTC).isoformat()
    model["corpus_name"] = args.corpus.name
    target = save_model(model, args.output)
    print(json.dumps({"saved": str(target), "sample_count": model["sample_count"], "selected_model": model.get("model_type"), "validation_metrics": model.get("validation_metrics"), "candidate_metrics": model.get("candidate_metrics")}, indent=2))


if __name__ == "__main__":
    main()
