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
        "When DOCX paragraph markers such as <<<P:12>>> and <<<ENDP:12>>> are present, keep every marker exactly unchanged and in the same order. Do not merge or split marked paragraphs. "
        "Return only the revised passage. Never add analysis or markdown fences.\n\nRules:\n" + rules
    )


def _refine_openai(text: str, config: RefinerConfig, mode: str = "balanced", *, context_before: str = "", context_after: str = "") -> str:
    """Refine with OpenAI's Responses API, recommended for GPT-5.6."""
    payload = {
        "model": config.model,
        "reasoning": {"effort": "medium" if mode == "deep" else "low"},
        "instructions": _system_prompt(mode),
        "input": (
            "READ-ONLY CONTEXT BEFORE (do not rewrite or return):\n" + context_before +
            "\n\nTARGET PASSAGE TO REVISE (return this passage only):\n" + text +
            "\n\nREAD-ONLY CONTEXT AFTER (do not rewrite or return):\n" + context_after
        ),
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


def _refine_openai_compatible(text: str, config: RefinerConfig, mode: str = "balanced", *, context_before: str = "", context_after: str = "") -> str:
    payload = {
        "model": config.model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user", "content": ("READ-ONLY CONTEXT BEFORE (do not rewrite or return):\n" + context_before + "\n\nTARGET PASSAGE TO REVISE (return this passage only):\n" + text + "\n\nREAD-ONLY CONTEXT AFTER (do not rewrite or return):\n" + context_after)},
        ],
    }
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    data = _post_json(f"{config.base_url}/chat/completions", payload, headers, config.timeout_seconds)
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RefinerError("The model service returned an unexpected response.") from exc


def _refine_ollama(text: str, config: RefinerConfig, mode: str = "balanced", *, context_before: str = "", context_after: str = "") -> str:
    payload = {
        "model": config.model,
        "stream": False,
        "options": {"temperature": 0.25},
        "messages": [
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user", "content": ("READ-ONLY CONTEXT BEFORE (do not rewrite or return):\n" + context_before + "\n\nTARGET PASSAGE TO REVISE (return this passage only):\n" + text + "\n\nREAD-ONLY CONTEXT AFTER (do not rewrite or return):\n" + context_after)},
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
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run Engine 2 as a resumable, section-aware protected batch job.

    Long manuscripts are split at scholarly section/paragraph boundaries. A
    compact previous/next context window is supplied read-only, while only the
    current batch may be rewritten. Each completed batch is reported through a
    checkpoint callback so the job can persist progress and retry only failures.
    """
    progress_callback = progress_callback or (lambda _pct, _stage: None)
    checkpoint_callback = checkpoint_callback or (lambda _state: None)
    cfg = config or RefinerConfig.from_env()
    public_level = "v2"
    if cfg.provider == "openai":
        cfg.model, public_level = resolve_engine2_model(model_override, cfg.model)
    status = provider_status(cfg)
    if not status["configured"]:
        return text, {"applied": False, "engine": "engine2", "label": "Engine 2, API rewrite", "level": public_level, "reason": "Engine 2 is not available on this deployment.", "batches": []}

    # v2.4.3 long-document defaults: 3.5k-5k semantic batches and up to three
    # concurrent remote requests. These can still be tuned for API rate limits.
    batch_words = max(2500, min(5000, int(os.getenv("HUMANIZER_ENGINE2_BATCH_WORDS", "4200"))))
    batches = build_humanizer_batches(text, max_words=batch_words)
    max_ratio = float(humanizer_variation_profile()["model_word_change_limit"])
    parallelism = max(1, min(4, int(os.getenv("HUMANIZER_ENGINE2_PARALLELISM", "3"))))
    retries = max(0, min(3, int(os.getenv("HUMANIZER_ENGINE2_BATCH_RETRIES", "2"))))
    total_words = sum(int(batch.get("word_count") or 0) for batch in batches) or len(text.split())
    progress_callback(5, f"Preparing {len(batches)} section-aware API batch(es) for {total_words:,} words…")

    def process_one(index: int, batch: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
        batch_text = str(batch["text"])
        if batch.get("protected"):
            return index, batch_text, {"index": index, "protected": True, "applied": False, "word_count": int(batch.get("word_count") or 0), "section_heading": batch.get("section_heading", "")}
        score = int((batch.get("diagnostic") or {}).get("naturalness_score", 0))
        should_refine = mode == "deep" or score < 92
        if not should_refine:
            return index, batch_text, {"index": index, "protected": False, "applied": False, "score": score, "word_count": int(batch.get("word_count") or 0), "section_heading": batch.get("section_heading", ""), "reason": "Already above selective threshold"}

        last_error = ""
        for attempt in range(1, retries + 2):
            try:
                kwargs = {
                    "context_before": str(batch.get("context_before") or ""),
                    "context_after": str(batch.get("context_after") or ""),
                }
                if cfg.provider == "ollama":
                    candidate = _refine_ollama(batch_text, cfg, mode, **kwargs)
                elif cfg.provider == "openai":
                    candidate = _refine_openai(batch_text, cfg, mode, **kwargs)
                elif cfg.provider == "openai_compatible":
                    candidate = _refine_openai_compatible(batch_text, cfg, mode, **kwargs)
                else:
                    raise RefinerError(f"Unsupported provider: {cfg.provider}")
                valid, issues = validate_humanizer_preservation(batch_text, candidate, max_word_change_ratio=max_ratio)
                if not valid:
                    last_error = "Preservation check failed: " + ", ".join(issues)
                    if attempt <= retries:
                        continue
                    return index, batch_text, {"index": index, "applied": False, "score_before": score, "score_after": score, "naturalness_gain": 0, "preservation_issues": issues, "word_count": int(batch.get("word_count") or 0), "section_heading": batch.get("section_heading", ""), "attempts": attempt, "fallback_retained": True}
                candidate_score = int(analyse_scholarly_style(candidate).get("naturalness_score", 0))
                if candidate_score < score:
                    last_error = "API candidate reduced naturalness."
                    if attempt <= retries:
                        continue
                    return index, batch_text, {"index": index, "applied": False, "score_before": score, "score_after": score, "naturalness_gain": 0, "preservation_issues": [], "word_count": int(batch.get("word_count") or 0), "section_heading": batch.get("section_heading", ""), "attempts": attempt, "reason": last_error, "fallback_retained": True}
                return index, candidate, {
                    "index": index, "applied": candidate != batch_text, "score_before": score,
                    "score_after": candidate_score, "naturalness_gain": candidate_score - score,
                    "preservation_issues": [], "word_count": int(batch.get("word_count") or 0),
                    "section_heading": batch.get("section_heading", ""), "attempts": attempt,
                }
            except RefinerError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = f"Engine 2 batch failed safely: {exc}"
            if attempt <= retries:
                continue
        return index, batch_text, {"index": index, "applied": False, "score": score, "error": last_error or "Engine 2 batch failed.", "word_count": int(batch.get("word_count") or 0), "section_heading": batch.get("section_heading", ""), "attempts": retries + 1, "fallback_retained": True}

    results: dict[int, tuple[str, dict[str, Any]]] = {}
    completed = 0
    completed_words = 0
    total = max(1, len(batches))

    def record(idx: int, output: str, report: dict[str, Any]) -> None:
        nonlocal completed, completed_words
        results[idx] = (output, report)
        completed += 1
        completed_words += int(report.get("word_count") or batches[idx].get("word_count") or 0)
        pct = int(8 + (completed_words / max(1, total_words)) * 86)
        stage = f"Engine 2: {completed}/{total} batches, {completed_words:,}/{total_words:,} words processed…"
        progress_callback(min(94, pct), stage)
        partial = []
        for batch_index, batch in enumerate(batches):
            partial.append(results.get(batch_index, (str(batch.get("text") or ""), {}))[0])
        checkpoint_callback({
            "completed_batches": completed, "total_batches": total,
            "completed_words": completed_words, "total_words": total_words,
            "last_batch": idx, "last_report": report,
            "partial_text": "\n\n".join(partial).strip(),
        })

    if parallelism == 1 or len(batches) <= 1:
        for index, batch in enumerate(batches):
            idx, output, report = process_one(index, batch)
            record(idx, output, report)
    else:
        with ThreadPoolExecutor(max_workers=min(parallelism, len(batches)), thread_name_prefix="engine2-api") as pool:
            future_map = {pool.submit(process_one, index, batch): index for index, batch in enumerate(batches)}
            for future in as_completed(future_map):
                idx, output, report = future.result()
                record(idx, output, report)

    ordered = [results[index] for index in range(len(batches))]
    outputs = [item[0] for item in ordered]
    batch_reports = [item[1] for item in ordered]
    revised = "\n\n".join(outputs).strip()
    failures = sum(1 for report in batch_reports if report.get("error") or report.get("fallback_retained") and not report.get("applied"))
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
        "batch_target_words": batch_words,
        "total_words": total_words,
        "batch_retries": retries,
        "section_aware": True,
        "context_continuity": True,
    }



_PARAGRAPH_START_RE = __import__("re").compile(r"(?m)^<<<P:(\d+)>>>\s*$")
_PARAGRAPH_BLOCK_RE = __import__("re").compile(r"(?ms)^<<<P:(\d+)>>>\s*\n(.*?)\n<<<ENDP:\1>>>\s*$")


def _encode_paragraph_group(group: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"<<<P:{int(item['paragraph_index'])}>>>\n{str(item.get('text') or '').strip()}\n<<<ENDP:{int(item['paragraph_index'])}>>>"
        for item in group
    )


def _parse_paragraph_group(candidate: str, expected_ids: list[int]) -> dict[int, str] | None:
    import re
    pattern = re.compile(r"(?ms)^<<<P:(\d+)>>>\s*\n(.*?)\n<<<ENDP:\1>>>\s*(?=\n\n<<<P:|\Z)")
    found: list[tuple[int, str]] = [(int(match.group(1)), match.group(2).strip()) for match in pattern.finditer(str(candidate or '').strip())]
    if [item[0] for item in found] != list(expected_ids):
        return None
    return {idx: text for idx, text in found}


def refine_paragraphs_with_model(
    paragraphs: list[dict[str, Any]],
    *,
    mode: str = "balanced",
    config: RefinerConfig | None = None,
    model_override: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Refine a DOCX paragraph map without flattening the Word document.

    Paragraph IDs are sent in 2.5k-5k word groups. The remote model must return
    the same marker sequence, after which each paragraph is independently passed
    through the existing evidence-preservation and naturalness gates. Failed
    paragraphs retain their source text; failed batches do not damage the DOCX.
    """
    progress_callback = progress_callback or (lambda _pct, _stage: None)
    checkpoint_callback = checkpoint_callback or (lambda _state: None)
    cfg = config or RefinerConfig.from_env()
    public_level = "v2"
    if cfg.provider == "openai":
        cfg.model, public_level = resolve_engine2_model(model_override, cfg.model)
    status = provider_status(cfg)
    originals = {int(item["paragraph_index"]): str(item.get("text") or "") for item in paragraphs}
    if not status["configured"]:
        return originals, {"applied": False, "engine": "engine2", "label": "Engine 2, API rewrite", "level": public_level, "reason": "Engine 2 is not available on this deployment.", "batches": []}

    batch_words = max(2500, min(5000, int(os.getenv("HUMANIZER_ENGINE2_BATCH_WORDS", "4200"))))
    parallelism = max(1, min(4, int(os.getenv("HUMANIZER_ENGINE2_PARALLELISM", "3"))))
    retries = max(0, min(3, int(os.getenv("HUMANIZER_ENGINE2_BATCH_RETRIES", "2"))))
    max_ratio = float(humanizer_variation_profile()["model_word_change_limit"])

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    for item in paragraphs:
        words = max(1, len(str(item.get("text") or "").split()))
        if current and current_words + words > batch_words:
            groups.append(current)
            current, current_words = [], 0
        current.append(item)
        current_words += words
    if current:
        groups.append(current)

    total_words = sum(max(1, len(text.split())) for text in originals.values())
    progress_callback(5, f"Preparing {len(groups)} format-preserving paragraph batch(es) for {total_words:,} editable words…")

    def process_group(index: int, group: list[dict[str, Any]]) -> tuple[int, dict[int, str], dict[str, Any]]:
        encoded = _encode_paragraph_group(group)
        expected_ids = [int(item["paragraph_index"]) for item in group]
        context_before = str(group[0].get("context_before") or "") if group else ""
        context_after = str(group[-1].get("context_after") or "") if group else ""
        last_error = ""
        for attempt in range(1, retries + 2):
            try:
                kwargs = {"context_before": context_before, "context_after": context_after}
                if cfg.provider == "ollama":
                    candidate = _refine_ollama(encoded, cfg, mode, **kwargs)
                elif cfg.provider == "openai":
                    candidate = _refine_openai(encoded, cfg, mode, **kwargs)
                elif cfg.provider == "openai_compatible":
                    candidate = _refine_openai_compatible(encoded, cfg, mode, **kwargs)
                else:
                    raise RefinerError(f"Unsupported provider: {cfg.provider}")
                parsed = _parse_paragraph_group(candidate, expected_ids)
                if parsed is None:
                    last_error = "Paragraph marker sequence changed; the batch was rejected."
                    if attempt <= retries:
                        continue
                    break
                accepted: dict[int, str] = {}
                rejected = 0
                for item in group:
                    pid = int(item["paragraph_index"])
                    source_text = originals[pid]
                    proposed = str(parsed.get(pid, source_text))
                    paragraph_limit = max(max_ratio, 0.35 if len(source_text.split()) < 120 else max_ratio)
                    valid, issues = validate_humanizer_preservation(source_text, proposed, max_word_change_ratio=paragraph_limit)
                    source_score = int(analyse_scholarly_style(source_text).get("naturalness_score", 0))
                    proposed_score = int(analyse_scholarly_style(proposed).get("naturalness_score", 0)) if valid else source_score
                    if not valid or proposed_score < source_score:
                        accepted[pid] = source_text
                        rejected += 1
                    else:
                        accepted[pid] = proposed
                return index, accepted, {
                    "index": index,
                    "applied": any(accepted[pid] != originals[pid] for pid in expected_ids),
                    "paragraphs": len(group),
                    "rejected_paragraphs": rejected,
                    "word_count": sum(max(1, len(originals[pid].split())) for pid in expected_ids),
                    "attempts": attempt,
                }
            except RefinerError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = f"Engine 2 format-preserving batch failed safely: {exc}"
            if attempt <= retries:
                continue
        fallback = {pid: originals[pid] for pid in expected_ids}
        return index, fallback, {
            "index": index, "applied": False, "paragraphs": len(group),
            "word_count": sum(max(1, len(originals[pid].split())) for pid in expected_ids),
            "attempts": retries + 1, "error": last_error or "Batch failed.", "fallback_retained": True,
        }

    results: dict[int, tuple[dict[int, str], dict[str, Any]]] = {}
    completed = 0
    completed_words = 0

    def record(index: int, mapping: dict[int, str], report: dict[str, Any]) -> None:
        nonlocal completed, completed_words
        results[index] = (mapping, report)
        completed += 1
        completed_words += int(report.get("word_count") or 0)
        pct = int(8 + (completed_words / max(1, total_words)) * 86)
        progress_callback(min(94, pct), f"Engine 2: {completed}/{len(groups)} Word batches, {completed_words:,}/{total_words:,} editable words processed…")
        partial: dict[int, str] = dict(originals)
        for mapping, _report in results.values():
            partial.update(mapping)
        checkpoint_callback({
            "completed_batches": completed, "total_batches": len(groups),
            "completed_words": completed_words, "total_words": total_words,
            "format_preserving_docx": True, "completed_paragraphs": sum(len(mapping) for mapping, _ in results.values()),
            "partial_paragraphs": partial,
        })

    if parallelism == 1 or len(groups) <= 1:
        for index, group in enumerate(groups):
            idx, mapping, report = process_group(index, group)
            record(idx, mapping, report)
    else:
        with ThreadPoolExecutor(max_workers=min(parallelism, len(groups)), thread_name_prefix="engine2-docx") as pool:
            future_map = {pool.submit(process_group, index, group): index for index, group in enumerate(groups)}
            for future in as_completed(future_map):
                idx, mapping, report = future.result()
                record(idx, mapping, report)

    revised = dict(originals)
    batch_reports: list[dict[str, Any]] = []
    for index in range(len(groups)):
        mapping, report = results[index]
        revised.update(mapping)
        batch_reports.append(report)
    failures = sum(1 for report in batch_reports if report.get("error") or report.get("fallback_retained"))
    progress_callback(100, "Engine 2 format-preserving Word refinement finished…")
    return revised, {
        "applied": any(revised[key] != originals[key] for key in originals),
        "engine": "engine2", "label": "Engine 2, API rewrite", "level": public_level,
        "batches": batch_reports, "batch_count": len(batch_reports), "failed_batches": failures,
        "parallelism": min(parallelism, max(1, len(groups))), "batch_target_words": batch_words,
        "total_words": total_words, "batch_retries": retries, "section_aware": True,
        "context_continuity": True, "format_preserving_docx": True,
    }
