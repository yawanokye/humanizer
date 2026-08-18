from __future__ import annotations

"""Developer Benchmark Lab for provenance-known detector calibration.

The lab is disabled by default. Enable it with HUMANIZER_BENCHMARK_LAB_ENABLED=true.
If HUMANIZER_DEVELOPER_TOKEN is set, write/train/evaluate operations require the
same token in the X-Developer-Token header. Benchmark files should live on a
persistent Render disk in production by setting HUMANIZER_BENCHMARK_DIR.
"""

import json
import os
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.calibration import (
    classification_metrics,
    load_model,
    predict_probability_with_model,
    save_model,
    train_best_meta_classifier,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIR = BASE_DIR / "calibration"

ALLOWED_PROVENANCE = {
    "human_original",
    "human_edited",
    "gpt_generated",
    "claude_generated",
    "gemini_generated",
    "other_ai_generated",
    "ai_human_edited",
    "human_ai_edited",
    "engine1_output",
    "engine2_output",
    "engine3_output",
    "other_humanizer_output",
}


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def benchmark_enabled() -> bool:
    return _truthy("HUMANIZER_BENCHMARK_LAB_ENABLED")


def benchmark_dir() -> Path:
    raw = os.getenv("HUMANIZER_BENCHMARK_DIR", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DIR


def corpus_path() -> Path:
    return benchmark_dir() / "benchmark_lab.jsonl"


def validation_path() -> Path:
    return benchmark_dir() / "validation_report.json"


def developer_token_configured() -> bool:
    return bool(os.getenv("HUMANIZER_DEVELOPER_TOKEN", "").strip())


def verify_developer_token(token: str | None) -> bool:
    expected = os.getenv("HUMANIZER_DEVELOPER_TOKEN", "").strip()
    if not benchmark_enabled():
        return False
    if not expected:
        # Explicitly enabling the lab without a token is allowed for local/private
        # deployments, but the status endpoint calls this out as a warning.
        return True
    return bool(token) and token == expected


def _label_from_provenance(provenance: str) -> int:
    return 0 if provenance in {"human_original", "human_edited"} else 1


def _safe_external_scores(value: dict[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in (value or {}).items():
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= score <= 100:
            result[str(key).strip().lower()] = round(score, 2)
    return result


def add_sample(
    *,
    text: str,
    provenance: str,
    source_family: str = "unknown",
    discipline: str = "unknown",
    document_type: str = "unknown",
    editing_level: str = "none",
    external_scores: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if provenance not in ALLOWED_PROVENANCE:
        raise ValueError(f"Unknown provenance '{provenance}'.")
    clean = str(text or "").strip()
    if len(clean.split()) < 40:
        raise ValueError("Benchmark samples should contain at least 40 words.")
    row = {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "label": _label_from_provenance(provenance),
        "provenance": provenance,
        "source_family": str(source_family or "unknown").strip()[:80],
        "discipline": str(discipline or "unknown").strip()[:80],
        "document_type": str(document_type or "unknown").strip()[:80],
        "editing_level": str(editing_level or "none").strip()[:80],
        "external_scores": _safe_external_scores(external_scores),
        "notes": str(notes or "").strip()[:500],
        "word_count": len(clean.split()),
        "text": clean,
    }
    path = corpus_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_samples() -> list[dict[str, Any]]:
    path = corpus_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and str(item.get("text") or "").strip() and int(item.get("label", -1)) in {0, 1}:
            rows.append(item)
    return rows


def _external_disagreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spreads: list[float] = []
    detector_counts: Counter[str] = Counter()
    for row in rows:
        scores = [float(v) for v in (row.get("external_scores") or {}).values()]
        detector_counts.update((row.get("external_scores") or {}).keys())
        if len(scores) >= 2:
            spreads.append(max(scores) - min(scores))
    return {
        "samples_with_multiple_external_scores": len(spreads),
        "mean_score_range": round(statistics.mean(spreads), 2) if spreads else None,
        "median_score_range": round(statistics.median(spreads), 2) if spreads else None,
        "detector_counts": dict(detector_counts),
    }


def corpus_status() -> dict[str, Any]:
    rows = load_samples()
    labels = Counter(int(row["label"]) for row in rows)
    return {
        "enabled": benchmark_enabled(),
        "token_configured": developer_token_configured(),
        "storage_path": str(corpus_path()),
        "sample_count": len(rows),
        "human_count": labels.get(0, 0),
        "ai_count": labels.get(1, 0),
        "by_provenance": dict(Counter(str(row.get("provenance") or "unknown") for row in rows)),
        "by_source_family": dict(Counter(str(row.get("source_family") or "unknown") for row in rows)),
        "by_discipline": dict(Counter(str(row.get("discipline") or "unknown") for row in rows)),
        "by_document_type": dict(Counter(str(row.get("document_type") or "unknown") for row in rows)),
        "by_editing_level": dict(Counter(str(row.get("editing_level") or "unknown") for row in rows)),
        "external_detector_disagreement": _external_disagreement(rows),
        "ready_to_train": len(rows) >= 40 and labels.get(0, 0) >= 20 and labels.get(1, 0) >= 20,
        "warning": "Benchmark Lab is enabled without a developer token. Use only on a private/local deployment." if benchmark_enabled() and not developer_token_configured() else "",
    }


def _feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Local import avoids an analyzer -> calibration -> benchmark circular import.
    from services.analyzer import dashboard_report

    result: list[dict[str, Any]] = []
    for row in rows:
        report = dashboard_report(str(row["text"]), use_calibrator=False)
        result.append({
            "id": row.get("id"),
            "label": int(row["label"]),
            "features": report.get("calibration_features") or {},
            "metadata": {
                "provenance": row.get("provenance"),
                "source_family": row.get("source_family"),
                "discipline": row.get("discipline"),
                "document_type": row.get("document_type"),
                "editing_level": row.get("editing_level"),
                "word_count": row.get("word_count"),
            },
        })
    return result


def train_from_benchmark() -> dict[str, Any]:
    rows = load_samples()
    feature_rows = _feature_rows(rows)
    model = train_best_meta_classifier(feature_rows)
    model["trained_at"] = datetime.now(UTC).isoformat()
    model["corpus_name"] = corpus_path().name
    target = save_model(model)
    corpus_evaluation = evaluate_model(model, rows)
    validation = {
        **corpus_evaluation,
        "overall": model.get("validation_metrics") or {},
        "heldout_selected_metrics": model.get("validation_metrics") or {},
        "corpus_fit_metrics": corpus_evaluation.get("overall") or {},
        "selected_model": model.get("model_type"),
        "candidate_metrics": model.get("candidate_metrics") or {},
        "model_path": str(target),
        "note": "Headline Validation Centre metrics come from the stratified held-out model-selection split. Group breakdowns below are descriptive full-corpus diagnostics and must not be presented as independent held-out performance.",
    }
    validation_path().write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return {"model": model, "validation": validation, "saved": str(target)}


def _group_metrics(rows: list[dict[str, Any]], probabilities: list[float], key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, prob in zip(rows, probabilities):
        buckets[str(row.get(key) or "unknown")].append((int(row["label"]), prob))
    result = {}
    for name, values in buckets.items():
        labels = [y for y, _ in values]
        if len(values) >= 4 and len(set(labels)) >= 2:
            result[name] = classification_metrics(labels, [p for _, p in values])
    return result


def _word_band(word_count: int) -> str:
    if word_count < 250:
        return "<250"
    if word_count <= 1000:
        return "250-1000"
    return ">1000"


def evaluate_model(model: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    active = model or load_model()
    if not active:
        raise ValueError("No trained calibration model is installed.")
    samples = rows if rows is not None else load_samples()
    feature_rows = _feature_rows(samples)
    probabilities = [float(predict_probability_with_model(active, item["features"])["probability"]) for item in feature_rows]
    labels = [int(row["label"]) for row in samples]
    enriched = []
    for row in samples:
        item = dict(row)
        item["word_band"] = _word_band(int(row.get("word_count") or len(str(row.get("text") or "").split())))
        enriched.append(item)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_count": len(samples),
        "overall": classification_metrics(labels, probabilities),
        "by_provenance": _group_metrics(enriched, probabilities, "provenance"),
        "by_source_family": _group_metrics(enriched, probabilities, "source_family"),
        "by_discipline": _group_metrics(enriched, probabilities, "discipline"),
        "by_document_type": _group_metrics(enriched, probabilities, "document_type"),
        "by_editing_level": _group_metrics(enriched, probabilities, "editing_level"),
        "by_word_count": _group_metrics(enriched, probabilities, "word_band"),
        "external_detector_disagreement": _external_disagreement(samples),
        "note": "Metrics are valid held-out evidence only when the evaluated samples were not used for training or model selection. The selected-model validation metrics in calibration_status come from v2.3's internal held-out split.",
    }


def validation_status() -> dict[str, Any]:
    path = validation_path()
    if not path.exists():
        return {"available": False, "message": "No Validation Centre report has been generated yet."}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "message": "Validation report could not be read."}
    return {"available": True, **report}
