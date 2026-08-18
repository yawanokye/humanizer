from __future__ import annotations

"""Optional true token-probability diagnostics using a local causal language model.

This module is deliberately optional because PyTorch/Transformers can exceed the
memory footprint of a small Render service.  The base application continues to
run without those packages.  Enable explicitly with ``REFERENCE_LM_ENABLED=true``.
"""

import math
import os
from typing import Any

_MODEL = None
_TOKENIZER = None
_LOAD_ERROR: str | None = None


def _enabled() -> bool:
    return os.getenv("REFERENCE_LM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def reference_lm_status() -> dict[str, Any]:
    model_name = os.getenv("REFERENCE_LM_MODEL", "distilgpt2").strip() or "distilgpt2"
    if not _enabled():
        return {
            "enabled": False,
            "available": False,
            "model": model_name,
            "mode": "disabled",
            "message": "True reference-LM perplexity is disabled. Statistical proxy metrics remain active.",
        }
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "model": model_name,
            "mode": "dependency-missing",
            "message": f"Reference LM enabled but optional dependencies are unavailable: {type(exc).__name__}.",
        }
    return {
        "enabled": True,
        "available": _LOAD_ERROR is None,
        "model": model_name,
        "mode": "ready" if _LOAD_ERROR is None else "load-failed",
        "message": "Reference-LM token probability diagnostics are enabled." if _LOAD_ERROR is None else _LOAD_ERROR,
    }


def _load():
    global _MODEL, _TOKENIZER, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL
    if _LOAD_ERROR:
        raise RuntimeError(_LOAD_ERROR)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = os.getenv("REFERENCE_LM_MODEL", "distilgpt2").strip() or "distilgpt2"
        local_only = os.getenv("REFERENCE_LM_LOCAL_ONLY", "true").strip().lower() not in {"0", "false", "no", "off"}
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_only)
        _MODEL.eval()
        # Keep CPU inference deterministic and avoid retaining gradients.
        for parameter in _MODEL.parameters():
            parameter.requires_grad_(False)
        return _TOKENIZER, _MODEL
    except Exception as exc:  # optional subsystem must never crash the app
        _LOAD_ERROR = f"Reference LM could not be loaded: {type(exc).__name__}: {exc}"
        raise RuntimeError(_LOAD_ERROR) from exc


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = min(1.0, max(0.0, q)) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def score_reference_lm(text: str) -> dict[str, Any]:
    status = reference_lm_status()
    if not status.get("enabled"):
        return {**status, "scored": False}
    try:
        import torch
        tokenizer, model = _load()
        max_tokens = max(128, min(8192, int(os.getenv("REFERENCE_LM_MAX_TOKENS", "2048"))))
        encoded = tokenizer(str(text or ""), return_tensors="pt", truncation=True, max_length=max_tokens)
        input_ids = encoded.get("input_ids")
        if input_ids is None or int(input_ids.shape[-1]) < 3:
            return {**status, "scored": False, "message": "Not enough tokens for reference-LM scoring."}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
            surprisals = (-token_log_probs).detach().cpu().tolist()
        mean = sum(surprisals) / len(surprisals)
        variance = sum((x - mean) ** 2 for x in surprisals) / len(surprisals)
        std = variance ** 0.5
        perplexity = math.exp(min(20.0, mean))
        low_threshold = float(os.getenv("REFERENCE_LM_LOW_SURPRISAL_THRESHOLD", "1.5"))
        low_flags = [x <= low_threshold for x in surprisals]
        longest = run = 0
        for flag in low_flags:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return {
            "enabled": True,
            "available": True,
            "scored": True,
            "mode": "reference-language-model",
            "model": os.getenv("REFERENCE_LM_MODEL", "distilgpt2").strip() or "distilgpt2",
            "token_count": len(surprisals),
            "perplexity": round(perplexity, 4),
            "surprisal_mean": round(mean, 4),
            "surprisal_std": round(std, 4),
            "surprisal_p10": round(_percentile(surprisals, 0.10), 4),
            "surprisal_p50": round(_percentile(surprisals, 0.50), 4),
            "surprisal_p90": round(_percentile(surprisals, 0.90), 4),
            "low_surprisal_share": round(sum(low_flags) / len(low_flags), 4),
            "longest_low_surprisal_run": int(longest),
            "message": "True token probabilities were measured with the configured reference language model. Interpret against a labelled calibration corpus, not as a stand-alone AI verdict.",
        }
    except Exception as exc:
        return {
            **reference_lm_status(),
            "available": False,
            "scored": False,
            "mode": "load-failed",
            "message": str(exc),
        }
