from __future__ import annotations

import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.client import RemoteDisconnected
from typing import Any, Callable
from urllib import error, request

from scholarly_humanizer import (
    analyse_scholarly_style,
    build_humanizer_batches,
    humanizer_variation_profile,
    scholarly_humanizer_prompt_rules,
    validate_humanizer_preservation,
)


@dataclass(slots=True)
class RefinerConfig:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: int = 150

    @classmethod
    def from_env(cls) -> "RefinerConfig":
        provider_raw = os.getenv("HUMANIZER_PROVIDER", "").strip().lower()
        # If a standard OpenAI key is present, Engine 2 should work even when
        # HUMANIZER_PROVIDER was omitted in an older Render deployment.
        provider = provider_raw or ("openai" if os.getenv("OPENAI_API_KEY", "").strip() else "none")

        # Engine 2 supports standard OpenAI environment variables directly.
        # Generic HUMANIZER_* names remain supported for backward compatibility
        # and for other OpenAI-compatible providers.
        if provider == "openai":
            model = (os.getenv("OPENAI_MODEL") or os.getenv("HUMANIZER_MODEL") or "gpt-5.6-terra").strip()
            base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("HUMANIZER_API_KEY", "")).strip()
        elif provider == "ollama":
            model = os.getenv("HUMANIZER_MODEL", "").strip()
            base_url = os.getenv("HUMANIZER_BASE_URL", "http://localhost:11434").rstrip("/")
            api_key = ""
        else:
            model = os.getenv("HUMANIZER_MODEL", "").strip()
            base_url = os.getenv("HUMANIZER_BASE_URL", "").rstrip("/")
            api_key = os.getenv("HUMANIZER_API_KEY", "").strip()

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=int(os.getenv("HUMANIZER_TIMEOUT_SECONDS", "150")),
        )


class RefinerError(RuntimeError):
    pass


ENGINE2_MODEL_ALIASES = {
    "v1": "gpt-5.6-luna",
    "v2": "gpt-5.6-terra",
}
ENGINE2_MODEL_PUBLIC_LABELS = {
    "v1": "V1 (Light)",
    "v2": "V2 (Moderate)",
}

def resolve_engine2_model(value: str | None, default_model: str) -> tuple[str, str]:
    """Resolve the public Engine 2 alias without exposing provider model names in the UI/API."""
    raw = str(value or "").strip()
    if raw in ENGINE2_MODEL_ALIASES:
        return ENGINE2_MODEL_ALIASES[raw], raw
    # Backward compatibility for saved clients from v2.3 and earlier.
    reverse = {model: alias for alias, model in ENGINE2_MODEL_ALIASES.items()}
    if raw in reverse:
        return raw, reverse[raw]
    if not raw:
        alias = reverse.get(default_model, "v2")
        return ENGINE2_MODEL_ALIASES.get(alias, default_model), alias
    raise RefinerError("Unsupported Engine 2 refinement level.")


def provider_status(config: RefinerConfig | None = None) -> dict[str, Any]:
    cfg = config or RefinerConfig.from_env()
    configured = (
        cfg.provider in {"ollama", "openai", "openai_compatible"}
        and bool(cfg.model and cfg.base_url)
        and (cfg.provider != "openai" or bool(cfg.api_key))
    )
    missing: list[str] = []
    if cfg.provider not in {"ollama", "openai", "openai_compatible"}:
        missing.append("HUMANIZER_PROVIDER=openai")
    if not cfg.model:
        missing.append("OPENAI_MODEL")
    if not cfg.base_url:
        missing.append("OPENAI_BASE_URL")
    if cfg.provider == "openai" and not cfg.api_key:
        missing.append("OPENAI_API_KEY")
    engine_2_message = (
        f"Engine 2 API rewrite is configured with {cfg.model}."
        if configured
        else "Engine 2 API rewrite is not configured" + (f". Missing: {', '.join(missing)}." if missing else ".")
    )
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "configured": configured,
        "missing": missing,
        "engines": {
            "engine1": {
                "label": "Engine 1, Local rewrite",
                "configured": True,
                "default": True,
                "uses_external_api": False,
                "message": "Engine 1 local rewrite is active. No API key is required.",
            },
            "engine2": {
                "label": "Engine 2, API rewrite",
                "configured": configured,
                "default": False,
                "uses_external_api": True,
                "provider": cfg.provider,
                "model": cfg.model,
                "supported_models": ["gpt-5.6-terra", "gpt-5.6-luna"] if cfg.provider == "openai" else [],
                "message": engine_2_message,
            },
        },
        "message": "Engine availability checked.",
    }


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RefinerError(f"Model service returned HTTP {exc.code}: {body[:500]}") from exc
    except error.URLError as exc:
        raise RefinerError(f"Could not reach the model service: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RefinerError(f"Model request timed out after {timeout} seconds. The unchanged protected text was retained for this batch.") from exc
    except (ConnectionResetError, RemoteDisconnected) as exc:
        raise RefinerError("The model service closed the connection before returning a complete response. The unchanged protected text was retained for this batch.") from exc
    except json.JSONDecodeError as exc:
        raise RefinerError("The model service returned an incomplete or invalid response. The unchanged protected text was retained for this batch.") from exc


def _system_prompt(mode: str = "balanced") -> str:
    rules = "\n".join(f"- {rule}" for rule in scholarly_humanizer_prompt_rules())
    depth = (
        "Apply a substantial line edit. Recast formulaic sentence openings, repetitive transitions, over-balanced structures, "
        "uniform cadence and generic metadiscourse wherever this can be done without changing substance."
        if mode == "deep"
        else "Apply a clear line edit to formulaic or repetitive prose while preserving the author’s disciplinary register."
    )
    return (
        "You are a preservation-gated scholarly style editor. Improve natural scholarly voice, flow, rhythm, specificity, and sentence variety. "
        + depth + " "
        "Do not insert typos, slang, fake uncertainty, invented examples or unsupported claims. "
        "Treat names, emails, dates, numbers, percentages, citations, references, URLs, DOIs, equations, tables, table headers, figure captions, headings, tickers and statistical notation as locked content. "
        "You may rewrite prose around those items, but do not alter, omit, invent, reorder or relabel any locked item. "
        "Return only the revised passage. Never add analysis or markdown fences.\n\nRules:\n" + rules
    )


def _refine_openai(text: str, config: RefinerConfig, mode: str = "balanced") -> str:
    """Refine with OpenAI's Responses API, recommended for GPT-5.6."""
    payload = {
        "model": config.model,
        "reasoning": {"effort": "medium" if mode == "deep" else "low"},
        "instructions": _system_prompt(mode),
        "input": "Revise this passage while preserving every fact, number, citation, heading and placeholder exactly:\n\n" + text,
        "store": False,
    }
    headers = {"Authorization": f"Bearer {config.api_key}"}
    data = _post_json(f"{config.base_url}/responses", payload, headers, config.timeout_seconds)

    # The raw Responses API returns output items containing output_text content.
    try:
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return str(content["text"]).strip()
    except (AttributeError, TypeError) as exc:
        raise RefinerError("OpenAI returned an unexpected Responses API payload.") from exc
    raise RefinerError("OpenAI returned no text output.")


def _refine_openai_compatible(text: str, config: RefinerConfig, mode: str = "balanced") -> str:
    payload = {
        "model": config.model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user", "content": "Revise this passage while preserving every fact, number, citation, heading and placeholder exactly:\n\n" + text},
        ],
    }
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    data = _post_json(f"{config.base_url}/chat/completions", payload, headers, config.timeout_seconds)
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RefinerError("The model service returned an unexpected response.") from exc


def _refine_ollama(text: str, config: RefinerConfig, mode: str = "balanced") -> str:
    payload = {
        "model": config.model,
        "stream": False,
        "options": {"temperature": 0.25},
        "messages": [
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user", "content": "Revise this passage while preserving every fact, number, citation, heading and placeholder exactly:\n\n" + text},
        ],
    }
    data = _post_json(f"{config.base_url}/api/chat", payload, {}, config.timeout_seconds)
    try:
        return str(data["message"]["content"]).strip()
    except (KeyError, TypeError) as exc:
        raise RefinerError("Ollama returned an unexpected response.") from exc


def refine_with_model(
    text: str,
    *,
    mode: str = "balanced",
    config: RefinerConfig | None = None,
    model_override: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run Engine 2 with preservation-gated batches.

    v2.4 can process independent API batches concurrently. This shortens wall-clock
    time for long manuscripts and, together with the job endpoint, avoids keeping a
    browser fetch open for the entire model rewrite.
    """
    progress_callback = progress_callback or (lambda _pct, _stage: None)
    cfg = config or RefinerConfig.from_env()
    public_level = "v2"
    if cfg.provider == "openai":
        cfg.model, public_level = resolve_engine2_model(model_override, cfg.model)
    status = provider_status(cfg)
    if not status["configured"]:
        return text, {"applied": False, "engine": "engine2", "label": "Engine 2, API rewrite", "level": public_level, "reason": "Engine 2 is not available on this deployment.", "batches": []}

    batch_words = max(700, min(3200, int(os.getenv("HUMANIZER_ENGINE2_BATCH_WORDS", "1800"))))
    batches = build_humanizer_batches(text, max_words=batch_words)
    max_ratio = float(humanizer_variation_profile()["model_word_change_limit"])
    parallelism = max(1, min(4, int(os.getenv("HUMANIZER_ENGINE2_PARALLELISM", "2"))))
    progress_callback(5, f"Preparing {len(batches)} protected API batch(es)…")

    def process_one(index: int, batch: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
        batch_text = str(batch["text"])
        if batch.get("protected"):
            return index, batch_text, {"index": index, "protected": True, "applied": False}
        score = int((batch.get("diagnostic") or {}).get("naturalness_score", 0))
        should_refine = mode == "deep" or score < 92
        if not should_refine:
            return index, batch_text, {"index": index, "protected": False, "applied": False, "score": score, "reason": "Already above selective threshold"}
        try:
            if cfg.provider == "ollama":
                candidate = _refine_ollama(batch_text, cfg, mode)
            elif cfg.provider == "openai":
                candidate = _refine_openai(batch_text, cfg, mode)
            elif cfg.provider == "openai_compatible":
                candidate = _refine_openai_compatible(batch_text, cfg, mode)
            else:
                raise RefinerError(f"Unsupported provider: {cfg.provider}")
            valid, issues = validate_humanizer_preservation(batch_text, candidate, max_word_change_ratio=max_ratio)
            if not valid:
                return index, batch_text, {"index": index, "applied": False, "score_before": score, "score_after": score, "naturalness_gain": 0, "preservation_issues": issues}
            candidate_score = int(analyse_scholarly_style(candidate).get("naturalness_score", 0))
            if candidate_score < score:
                return index, batch_text, {
                    "index": index, "applied": False, "score_before": score, "score_after": score,
                    "naturalness_gain": 0, "preservation_issues": [],
                    "reason": "API candidate was rejected because it reduced naturalness.",
                }
            return index, candidate, {
                "index": index, "applied": candidate != batch_text, "score_before": score,
                "score_after": candidate_score, "naturalness_gain": candidate_score - score,
                "preservation_issues": [],
            }
        except RefinerError as exc:
            return index, batch_text, {"index": index, "applied": False, "score": score, "error": str(exc), "fallback_retained": True}
        except Exception as exc:
            # A single model/network failure must not terminate the full document job.
            return index, batch_text, {"index": index, "applied": False, "score": score, "error": f"Engine 2 batch failed safely: {exc}", "fallback_retained": True}

    results: dict[int, tuple[str, dict[str, Any]]] = {}
    completed = 0
    total = max(1, len(batches))
    if parallelism == 1 or len(batches) <= 1:
        for index, batch in enumerate(batches):
            idx, output, report = process_one(index, batch)
            results[idx] = (output, report)
            completed += 1
            progress_callback(int(8 + (completed / total) * 86), f"Engine 2 batch {completed}/{total} completed…")
    else:
        with ThreadPoolExecutor(max_workers=min(parallelism, len(batches)), thread_name_prefix="engine2-api") as pool:
            future_map = {pool.submit(process_one, index, batch): index for index, batch in enumerate(batches)}
            for future in as_completed(future_map):
                idx, output, report = future.result()
                results[idx] = (output, report)
                completed += 1
                progress_callback(int(8 + (completed / total) * 86), f"Engine 2 batch {completed}/{total} completed…")

    ordered = [results[index] for index in range(len(batches))]
    outputs = [item[0] for item in ordered]
    batch_reports = [item[1] for item in ordered]
    revised = "\n\n".join(outputs).strip()
    failures = sum(1 for report in batch_reports if report.get("error"))
    progress_callback(100, "Engine 2 API refinement finished…")
    return revised, {
        "applied": revised != text,
        "engine": "engine2",
        "label": "Engine 2, API rewrite",
        "level": public_level,
        "batches": batch_reports,
        "batch_count": len(batch_reports),
        "failed_batches": failures,
        "parallelism": min(parallelism, max(1, len(batches))),
    }
