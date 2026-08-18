from __future__ import annotations

"""Corpus-calibrated meta-classification for AI-style signals.

v2.3 keeps a transparent fallback when no labelled corpus is installed, but can
train and compare several lightweight, explainable meta-classifiers without a
heavy ML dependency. Model selection is based on a stratified held-out split,
not in-sample fit. The selected family is then refit on the full benchmark while
retaining the held-out validation metrics that justified the choice.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "calibration" / "meta_classifier.json"

FEATURE_NAMES = (
    "forensic_pct",
    "statistical_pct",
    "paragraph_pct",
    "regularity_pct",
    "perplexity_proxy",
    "burstiness",
    "token_distribution_proxy",
    "ngram_frequency",
    "uniform_semantic_density",
    "repetitive_syntactic_structures",
    "systemic_transitions",
    "low_vocabulary_diversity",
    "segment_p90",
    "flagged_segment_ratio",
    "reference_perplexity",
    "reference_surprisal_mean",
    "reference_surprisal_std",
    "reference_low_surprisal_share",
    # v2.3 section-aware features. Missing sections resolve to zero.
    "section_abstract_mean",
    "section_intro_lit_mean",
    "section_methods_mean",
    "section_results_mean",
    "section_discussion_mean",
    "section_conclusion_mean",
    "section_other_mean",
    "section_score_spread",
)


def _model_path() -> Path:
    raw = os.getenv("HUMANIZER_CALIBRATION_MODEL", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MODEL_PATH


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(60.0, value))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-60.0, value))
    return z / (1.0 + z)


def _softmax_two(a: float, b: float) -> float:
    # Probability assigned to the second (AI) class.
    delta = max(-60.0, min(60.0, a - b))
    return 1.0 / (1.0 + math.exp(delta))


def load_model() -> dict[str, Any] | None:
    path = _model_path()
    if not path.exists():
        return None
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(model, dict) or not model.get("trained"):
        return None
    model_type = str(model.get("model_type") or "logistic")
    if model_type == "logistic" and not isinstance(model.get("weights"), dict):
        return None
    if model_type == "gaussian_nb" and not isinstance(model.get("classes"), dict):
        return None
    if model_type == "nearest_centroid" and not isinstance(model.get("centroids"), dict):
        return None
    return model


def calibration_status() -> dict[str, Any]:
    path = _model_path()
    model = load_model()
    if not model:
        return {
            "trained": False,
            "mode": "uncalibrated",
            "model_path": str(path),
            "sample_count": 0,
            "message": "No labelled calibration model is installed. The transparent four-layer ensemble is being used.",
        }
    metrics = model.get("validation_metrics") or model.get("metrics") or {}
    return {
        "trained": True,
        "mode": "trained-meta-classifier",
        "model_type": model.get("model_type", "logistic"),
        "model_path": str(path),
        "sample_count": int(model.get("sample_count") or 0),
        "positive_count": int(model.get("positive_count") or 0),
        "negative_count": int(model.get("negative_count") or 0),
        "metrics": metrics,
        "validation_metrics": model.get("validation_metrics") or {},
        "candidate_metrics": model.get("candidate_metrics") or {},
        "feature_count": len(model.get("feature_names") or model.get("weights") or {}),
        "trained_at": model.get("trained_at"),
        "corpus_name": model.get("corpus_name"),
        "message": "A labelled, held-out-validated meta-classifier is active. The headline AI Signal comes from the selected learned model.",
    }


def _standardised_vector(model: dict[str, Any], features: dict[str, float]) -> tuple[list[str], list[float]]:
    names = list(model.get("feature_names") or (model.get("weights") or {}).keys())
    means = model.get("means") or {}
    scales = model.get("scales") or {}
    values = []
    for name in names:
        x = float(features.get(name, 0.0) or 0.0)
        mean = float(means.get(name, 0.0) or 0.0)
        scale = max(1e-9, float(scales.get(name, 1.0) or 1.0))
        values.append((x - mean) / scale)
    return names, values


def predict_probability_with_model(model: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    model_type = str(model.get("model_type") or "logistic")
    names, vector = _standardised_vector(model, features)
    contributions: dict[str, float] = {}

    if model_type == "gaussian_nb":
        classes = model.get("classes") or {}
        scores: dict[str, float] = {}
        for class_key in ("0", "1"):
            info = classes.get(class_key) or {}
            prior = max(1e-9, float(info.get("prior") or 0.5))
            means = info.get("means") or {}
            variances = info.get("variances") or {}
            logp = math.log(prior)
            for name, x in zip(names, vector):
                mu = float(means.get(name, 0.0) or 0.0)
                var = max(1e-5, float(variances.get(name, 1.0) or 1.0))
                logp += -0.5 * (math.log(2 * math.pi * var) + ((x - mu) ** 2) / var)
            scores[class_key] = logp
        probability = _softmax_two(scores.get("0", 0.0), scores.get("1", 0.0))
        logit = math.log(max(1e-9, probability) / max(1e-9, 1 - probability))

    elif model_type == "nearest_centroid":
        centroids = model.get("centroids") or {}
        c0 = centroids.get("0") or {}
        c1 = centroids.get("1") or {}
        d0 = sum((x - float(c0.get(name, 0.0) or 0.0)) ** 2 for name, x in zip(names, vector))
        d1 = sum((x - float(c1.get(name, 0.0) or 0.0)) ** 2 for name, x in zip(names, vector))
        temperature = max(0.2, float(model.get("temperature") or 1.0))
        probability = _softmax_two(-d0 / temperature, -d1 / temperature)
        logit = math.log(max(1e-9, probability) / max(1e-9, 1 - probability))

    else:
        weights = model.get("weights") or {}
        value = float(model.get("bias") or 0.0)
        for name, x in zip(names, vector):
            weight = float(weights.get(name, 0.0) or 0.0)
            contribution = x * weight
            value += contribution
            contributions[name] = round(contribution, 4)
        probability = _sigmoid(value)
        logit = value

    return {
        "probability": probability,
        "percentage": round(probability * 100),
        "logit": round(logit, 4),
        "contributions": contributions,
        "model_type": model_type,
        "model_version": model.get("version", "1"),
        "sample_count": int(model.get("sample_count") or 0),
        "metrics": model.get("validation_metrics") or model.get("metrics") or {},
    }


def predict_probability(features: dict[str, float]) -> dict[str, Any] | None:
    model = load_model()
    if not model:
        return None
    return predict_probability_with_model(model, features)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if not values:
        return 1.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return max(1e-6, variance ** 0.5)


def _classification_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float]:
    if not labels:
        return {}
    preds = [1 if p >= threshold else 0 for p in probabilities]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(labels))
    positives = [p for p, y in zip(probabilities, labels) if y == 1]
    negatives = [p for p, y in zip(probabilities, labels) if y == 0]
    wins = ties = 0.0
    for pp in positives:
        for pn in negatives:
            if pp > pn:
                wins += 1
            elif pp == pn:
                ties += 1
    auc = (wins + 0.5 * ties) / max(1, len(positives) * len(negatives))
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fp / max(1, fp + tn), 4),
        "false_negative_rate": round(fn / max(1, fn + tp), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def classification_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float]:
    return _classification_metrics(labels, probabilities, threshold)


def _prepare(rows: Iterable[dict[str, Any]], *, min_samples: int = 40, min_per_class: int = 15) -> tuple[list[dict[str, Any]], list[int], list[str], dict[str, float], dict[str, float], list[list[float]]]:
    data = [row for row in rows if isinstance(row, dict) and int(row.get("label", -1)) in {0, 1}]
    labels = [int(row["label"]) for row in data]
    positives = sum(labels)
    negatives = len(labels) - positives
    if len(data) < min_samples or positives < min_per_class or negatives < min_per_class:
        raise ValueError(f"Calibration requires at least {min_samples} labelled samples and at least {min_per_class} samples in each class.")
    feature_names = [name for name in FEATURE_NAMES if any(name in (row.get("features") or {}) for row in data)]
    if len(feature_names) < 4:
        raise ValueError("Calibration rows do not contain enough detector features.")
    columns = {name: [float((row.get("features") or {}).get(name, 0.0) or 0.0) for row in data] for name in feature_names}
    means = {name: _mean(values) for name, values in columns.items()}
    scales = {name: _std(values, means[name]) for name, values in columns.items()}
    matrix = [[(float((row.get("features") or {}).get(name, 0.0) or 0.0) - means[name]) / scales[name] for name in feature_names] for row in data]
    return data, labels, feature_names, means, scales, matrix


def _fit_logistic(data: list[dict[str, Any]], *, epochs: int = 1600, learning_rate: float = 0.035, l2: float = 0.012) -> dict[str, Any]:
    data, labels, names, means, scales, matrix = _prepare(data, min_samples=20, min_per_class=8)
    positives = sum(labels)
    weights = [0.0] * len(names)
    base_rate = min(0.98, max(0.02, positives / len(labels)))
    bias = math.log(base_rate / (1 - base_rate))
    for _ in range(max(50, int(epochs))):
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        for x, y in zip(matrix, labels):
            p = _sigmoid(bias + sum(w * xi for w, xi in zip(weights, x)))
            error = p - y
            grad_b += error
            for j, xi in enumerate(x):
                grad_w[j] += error * xi
        bias -= learning_rate * grad_b / len(matrix)
        for j in range(len(weights)):
            weights[j] -= learning_rate * (grad_w[j] / len(matrix) + l2 * weights[j])
    return {
        "trained": True, "model_type": "logistic", "version": "2.3-logistic-1",
        "sample_count": len(data), "positive_count": positives, "negative_count": len(data) - positives,
        "feature_names": names, "means": means, "scales": scales,
        "weights": {name: weights[i] for i, name in enumerate(names)}, "bias": bias,
    }


def _fit_gaussian_nb(data: list[dict[str, Any]]) -> dict[str, Any]:
    data, labels, names, means, scales, matrix = _prepare(data, min_samples=20, min_per_class=8)
    classes: dict[str, Any] = {}
    for klass in (0, 1):
        rows = [x for x, y in zip(matrix, labels) if y == klass]
        class_means = {name: _mean([row[i] for row in rows]) for i, name in enumerate(names)}
        class_vars = {}
        for i, name in enumerate(names):
            mu = class_means[name]
            vals = [row[i] for row in rows]
            class_vars[name] = max(1e-3, sum((v - mu) ** 2 for v in vals) / max(1, len(vals)))
        classes[str(klass)] = {"prior": len(rows) / len(matrix), "means": class_means, "variances": class_vars}
    return {
        "trained": True, "model_type": "gaussian_nb", "version": "2.3-gnb-1",
        "sample_count": len(data), "positive_count": sum(labels), "negative_count": len(data) - sum(labels),
        "feature_names": names, "means": means, "scales": scales, "classes": classes,
    }


def _fit_centroid(data: list[dict[str, Any]]) -> dict[str, Any]:
    data, labels, names, means, scales, matrix = _prepare(data, min_samples=20, min_per_class=8)
    centroids: dict[str, dict[str, float]] = {}
    within_distances: list[float] = []
    for klass in (0, 1):
        rows = [x for x, y in zip(matrix, labels) if y == klass]
        center = [_mean([row[i] for row in rows]) for i in range(len(names))]
        centroids[str(klass)] = {name: center[i] for i, name in enumerate(names)}
        within_distances.extend(sum((row[i] - center[i]) ** 2 for i in range(len(names))) for row in rows)
    temperature = max(0.5, _mean(within_distances))
    return {
        "trained": True, "model_type": "nearest_centroid", "version": "2.3-centroid-1",
        "sample_count": len(data), "positive_count": sum(labels), "negative_count": len(data) - sum(labels),
        "feature_names": names, "means": means, "scales": scales, "centroids": centroids, "temperature": temperature,
    }


def train_logistic_meta_classifier(rows: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible direct logistic trainer."""
    data = list(rows)
    _prepare(data)
    model = _fit_logistic(data, **kwargs)
    labels = [int(row["label"]) for row in data if int(row.get("label", -1)) in {0, 1}]
    probs = [predict_probability_with_model(model, row.get("features") or {})["probability"] for row in data if int(row.get("label", -1)) in {0, 1}]
    model["metrics"] = _classification_metrics(labels, probs)
    model["training_note"] = "In-sample diagnostics only. v2.3 train_best_meta_classifier is preferred because it performs held-out model selection."
    return model


def _stratified_split(rows: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for row in rows:
        if int(row.get("label", -1)) in {0, 1}:
            by_class[int(row["label"])].append(row)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for klass in (0, 1):
        items = list(by_class[klass])
        rng.shuffle(items)
        n_valid = max(5, round(len(items) * validation_fraction))
        n_valid = min(n_valid, max(5, len(items) - 15))
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])
    rng.shuffle(train); rng.shuffle(valid)
    return train, valid


def train_best_meta_classifier(
    rows: Iterable[dict[str, Any]], *, validation_fraction: float = 0.25, seed: int = 42
) -> dict[str, Any]:
    data = [row for row in rows if isinstance(row, dict) and int(row.get("label", -1)) in {0, 1}]
    # Validate full corpus before split. Held-out model selection needs enough
    # examples to leave at least 15 per class in the training fold.
    _prepare(data)
    class_counts = {klass: sum(1 for row in data if int(row["label"]) == klass) for klass in (0, 1)}
    if min(class_counts.values()) < 20:
        raise ValueError("v2.3 held-out model selection requires at least 20 human and 20 AI/AI-edited samples.")
    train, valid = _stratified_split(data, validation_fraction, seed)
    if len({int(row["label"]) for row in valid}) < 2:
        raise ValueError("Held-out split needs both human and AI samples.")
    fitters = {"logistic": _fit_logistic, "gaussian_nb": _fit_gaussian_nb, "nearest_centroid": _fit_centroid}
    candidates: dict[str, dict[str, float]] = {}
    for name, fitter in fitters.items():
        model = fitter(train)
        labels = [int(row["label"]) for row in valid]
        probs = [float(predict_probability_with_model(model, row.get("features") or {})["probability"]) for row in valid]
        candidates[name] = _classification_metrics(labels, probs)
    def rank(item: tuple[str, dict[str, float]]) -> tuple[float, float, float, float]:
        _name, m = item
        return (float(m.get("roc_auc", 0)), float(m.get("f1", 0)), -float(m.get("false_positive_rate", 1)), float(m.get("accuracy", 0)))
    selected_name, selected_metrics = max(candidates.items(), key=rank)
    final = fitters[selected_name](data)
    final.update({
        "validation_metrics": selected_metrics,
        "candidate_metrics": candidates,
        "selection_method": "stratified held-out comparison by ROC-AUC, F1, false-positive rate and accuracy",
        "validation_fraction": validation_fraction,
        "validation_count": len(valid),
        "random_seed": seed,
        "version": f"2.3-{selected_name}-selected",
        "metrics": selected_metrics,
        "training_note": "Model family was selected on a stratified held-out split, then refit on the full benchmark. Reported validation metrics remain from the held-out selection split.",
    })
    return final


def save_model(model: dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path) if path else _model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    return target
