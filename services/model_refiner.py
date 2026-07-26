from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from scholarly_humanizer import (
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
    timeout_seconds: int = 180

    @classmethod
    def from_env(cls) -> "RefinerConfig":
        provider = os.getenv("HUMANIZER_PROVIDER", "none").strip().lower()
        default_url = "http://localhost:11434" if provider == "ollama" else ""
        return cls(
            provider=provider,
            model=os.getenv("HUMANIZER_MODEL", "").strip(),
            base_url=os.getenv("HUMANIZER_BASE_URL", default_url).rstrip("/"),
            api_key=os.getenv("HUMANIZER_API_KEY", "").strip(),
            timeout_seconds=int(os.getenv("HUMANIZER_TIMEOUT_SECONDS", "180")),
        )


class RefinerError(RuntimeError):
    pass


def provider_status(config: RefinerConfig | None = None) -> dict[str, Any]:
    cfg = config or RefinerConfig.from_env()
    configured = cfg.provider in {"ollama", "openai_compatible"} and bool(cfg.model and cfg.base_url)
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "configured": configured,
        "message": "Model refinement is available." if configured else "Local protected refinement is active. No external model is configured.",
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


def _system_prompt() -> str:
    rules = "\n".join(f"- {rule}" for rule in scholarly_humanizer_prompt_rules())
    return (
        "You are a preservation-gated scholarly style editor. Improve natural scholarly voice, flow, rhythm, and precision. "
        "Return only the revised passage. Never add analysis or markdown fences.\n\nRules:\n" + rules
    )


def _refine_openai_compatible(text: str, config: RefinerConfig) -> str:
    payload = {
        "model": config.model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": "Revise this passage while preserving every fact, number, citation, heading and placeholder exactly:\n\n" + text},
        ],
    }
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    data = _post_json(f"{config.base_url}/chat/completions", payload, headers, config.timeout_seconds)
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RefinerError("The model service returned an unexpected response.") from exc


def _refine_ollama(text: str, config: RefinerConfig) -> str:
    payload = {
        "model": config.model,
        "stream": False,
        "options": {"temperature": 0.25},
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": "Revise this passage while preserving every fact, number, citation, heading and placeholder exactly:\n\n" + text},
        ],
    }
    data = _post_json(f"{config.base_url}/api/chat", payload, {}, config.timeout_seconds)
    try:
        return str(data["message"]["content"]).strip()
    except (KeyError, TypeError) as exc:
        raise RefinerError("Ollama returned an unexpected response.") from exc


def refine_with_model(text: str, *, mode: str = "balanced", config: RefinerConfig | None = None) -> tuple[str, dict[str, Any]]:
    cfg = config or RefinerConfig.from_env()
    status = provider_status(cfg)
    if not status["configured"]:
        return text, {"applied": False, "provider": cfg.provider, "reason": status["message"], "batches": []}

    batches = build_humanizer_batches(text)
    outputs: list[str] = []
    batch_reports: list[dict[str, Any]] = []
    max_ratio = float(humanizer_variation_profile()["model_word_change_limit"])
    for index, batch in enumerate(batches):
        batch_text = str(batch["text"])
        if batch.get("protected"):
            outputs.append(batch_text)
            batch_reports.append({"index": index, "protected": True, "applied": False})
            continue
        score = int((batch.get("diagnostic") or {}).get("naturalness_score", 0))
        should_refine = mode == "deep" or score < 82
        if not should_refine:
            outputs.append(batch_text)
            batch_reports.append({"index": index, "protected": False, "applied": False, "score": score, "reason": "Already above selective threshold"})
            continue
        try:
            if cfg.provider == "ollama":
                candidate = _refine_ollama(batch_text, cfg)
            elif cfg.provider == "openai_compatible":
                candidate = _refine_openai_compatible(batch_text, cfg)
            else:
                raise RefinerError(f"Unsupported provider: {cfg.provider}")
            valid, issues = validate_humanizer_preservation(batch_text, candidate, max_word_change_ratio=max_ratio)
            if not valid:
                outputs.append(batch_text)
                batch_reports.append({"index": index, "applied": False, "score": score, "preservation_issues": issues})
            else:
                outputs.append(candidate)
                batch_reports.append({"index": index, "applied": candidate != batch_text, "score": score, "preservation_issues": []})
        except RefinerError as exc:
            outputs.append(batch_text)
            batch_reports.append({"index": index, "applied": False, "score": score, "error": str(exc)})
    revised = "\n\n".join(outputs).strip()
    return revised, {
        "applied": revised != text,
        "provider": cfg.provider,
        "model": cfg.model,
        "batches": batch_reports,
    }
