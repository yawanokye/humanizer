from __future__ import annotations

import html
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scholarly_humanizer import humanize_scholarly_text, humanize_signal_guided, preservation_certificate, validate_humanizer_preservation
from services.analyzer import dashboard_report
from services.calibration import calibration_status
from services.reference_lm import reference_lm_status
from services.benchmark import (
    ALLOWED_PROVENANCE, add_sample, benchmark_enabled, corpus_status, developer_token_configured, evaluate_model,
    train_from_benchmark, validation_status, verify_developer_token, registry_status,
    promote_model, rollback_model, drift_status,
)
from services.document_io import build_annotated_docx, build_docx, extract_text
from services.document_structure import inspect_docx, patch_docx, render_structured_text, text_digest, format_preservation_certificate
from services.model_refiner import provider_status, refine_with_model, refine_paragraphs_with_model

BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = int(os.getenv("HUMANIZER_MAX_INPUT_CHARS", "1200000"))
MAX_UPLOAD_BYTES = int(os.getenv("HUMANIZER_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Scholarly Humanizer",
    version="2.4.4",
    description="Format-preserving scholarly humanization and AI-style screening with protected DOCX patching, large-document jobs, private calibration, signal-coloured diagnostics, independent Engine 1, API Engine 2, and signal-guided Engine 3.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# v2.4 long-request resilience. Humanization can involve multiple external model
# calls and should not keep one browser HTTP request open for several minutes.
_HUMANIZE_JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("HUMANIZER_JOB_WORKERS", "1"))),
    thread_name_prefix="humanize-job",
)
_HUMANIZE_JOBS: dict[str, dict] = {}
_HUMANIZE_JOBS_LOCK = threading.Lock()
_HUMANIZE_JOB_TTL_SECONDS = max(600, int(os.getenv("HUMANIZER_JOB_TTL_SECONDS", "3600")))

# Uploaded DOCX packages are retained temporarily so a humanized Word export can
# patch the original OOXML instead of reconstructing the manuscript from plain
# text. This intentionally follows the single-process deployment used by v2.4.
_DOCUMENT_STORE: dict[str, dict] = {}
_DOCUMENT_STORE_LOCK = threading.Lock()
_DOCUMENT_TTL_SECONDS = max(1800, int(os.getenv("HUMANIZER_DOCUMENT_TTL_SECONDS", "14400")))
_DOCX_FORMAT_PRESERVATION = os.getenv("HUMANIZER_DOCX_FORMAT_PRESERVATION", "true").strip().lower() in {"1", "true", "yes", "on"}

def _clean_document_store() -> None:
    cutoff = time.time() - _DOCUMENT_TTL_SECONDS
    with _DOCUMENT_STORE_LOCK:
        stale = [doc_id for doc_id, item in _DOCUMENT_STORE.items() if float(item.get("updated_at", 0)) < cutoff]
        for doc_id in stale:
            _DOCUMENT_STORE.pop(doc_id, None)

def _get_document(document_id: str | None) -> dict | None:
    if not document_id:
        return None
    _clean_document_store()
    with _DOCUMENT_STORE_LOCK:
        item = _DOCUMENT_STORE.get(str(document_id))
        if item is not None:
            item["updated_at"] = time.time()
        return item

def _store_humanized_document(document_id: str, job_id: str, content: bytes, certificate: dict) -> None:
    with _DOCUMENT_STORE_LOCK:
        item = _DOCUMENT_STORE.get(document_id)
        if item is None:
            return
        outputs = item.setdefault("humanized_outputs", {})
        outputs[job_id] = {"content": content, "certificate": certificate, "created_at": time.time()}
        # Keep only the newest few exports per uploaded manuscript.
        if len(outputs) > 4:
            for old_job in sorted(outputs, key=lambda key: float(outputs[key].get("created_at", 0)))[:-4]:
                outputs.pop(old_job, None)
        item["updated_at"] = time.time()

def _clean_humanize_jobs() -> None:
    cutoff = time.time() - _HUMANIZE_JOB_TTL_SECONDS
    with _HUMANIZE_JOBS_LOCK:
        stale = [job_id for job_id, job in _HUMANIZE_JOBS.items() if float(job.get("updated_at", 0)) < cutoff and job.get("status") in {"completed", "failed"}]
        for job_id in stale:
            _HUMANIZE_JOBS.pop(job_id, None)

def _update_humanize_job(job_id: str, **changes) -> None:
    with _HUMANIZE_JOBS_LOCK:
        job = _HUMANIZE_JOBS.get(job_id)
        if not job:
            return
        job.update(changes)
        job["updated_at"] = time.time()

def _job_progress(job_id: str, progress: int, stage: str) -> None:
    _update_humanize_job(job_id, progress=max(0, min(99, int(progress))), stage=str(stage))

def _job_checkpoint(job_id: str, state: dict) -> None:
    """Persist completed Engine 2 batch state in the in-process job record.

    The partial manuscript is retained server-side while the job is active so a
    single failed batch does not discard completed work. Status polling exposes
    only the compact summary, not the manuscript text.
    """
    summary = {key: value for key, value in state.items() if key not in {"partial_text", "partial_paragraphs"}}
    _update_humanize_job(job_id, checkpoint=state, checkpoint_summary=summary)


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class HumanizeRequest(TextRequest):
    mode: Literal["light", "balanced", "deep"] = "balanced"
    engine: Literal["engine1", "engine2", "engine3"] = "engine1"
    engine2_model: Literal["v1", "v2", "gpt-5.6-terra", "gpt-5.6-luna"] = "v2"
    document_id: str | None = None
    # Backward-compatible field for older frontends. New UI uses engine.
    use_model: bool = False


class ExportRequest(TextRequest):
    title: str = "Scholarly Humanized Text"
    annotated: bool = False
    document_id: str | None = None
    humanize_job_id: str | None = None


class BenchmarkSampleRequest(TextRequest):
    provenance: str
    source_family: str = "unknown"
    discipline: str = "unknown"
    document_type: str = "unknown"
    editing_level: str = "none"
    external_scores: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class AdversarialAuditRequest(TextRequest):
    mode: Literal["light", "balanced", "deep"] = "deep"


class ModelRegistryRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=160)


def _validate_size(text: str) -> str:
    value = str(text or "")
    if len(value) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Input exceeds the {MAX_INPUT_CHARS:,}-character limit.",
        )
    return value


async def _read_upload_limited(file: UploadFile) -> bytes:
    """Read an upload without allowing an unbounded in-memory payload."""
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB upload limit.",
                )
            chunks.append(chunk)
    finally:
        await file.close()
    return b"".join(chunks)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    )
    if request.url.path.startswith("/api/") or request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    """Lightweight Render health check that does not call an external model."""
    return {"status": "ok", "version": app.version}


@app.get("/api/status")
def status() -> dict:
    """Public capability status. Detector internals stay behind developer access."""
    provider = provider_status()
    configured = bool(provider.get("configured"))
    return {
        "ok": True,
        "version": app.version,
        "engines": {
            "engine1": {"configured": True, "label": "Engine 1, Local rewrite"},
            "engine2": {"configured": configured, "label": "Engine 2, API rewrite"},
            "engine3": {"configured": True, "label": "Engine 3, Signal-Guided rewrite"},
        },
        "max_input_chars": MAX_INPUT_CHARS,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "developer_lab_available": benchmark_enabled() and developer_token_configured(),
    }


@app.get("/api/developer/detector/status")
def get_private_detector_status(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    return {
        "calibration": calibration_status(),
        "reference_lm": reference_lm_status(),
        "benchmark": corpus_status(),
        "validation": validation_status(),
        "registry": registry_status(),
    }

@app.post("/api/developer/analyse")
def developer_analyse(payload: TextRequest, x_developer_token: str | None = Header(default=None)) -> dict:
    """Private detector analysis including calibration/probability internals."""
    _require_benchmark_access(x_developer_token)
    return dashboard_report(_validate_size(payload.text), include_private=True)

def _require_benchmark_access(token: str | None) -> None:
    if not benchmark_enabled():
        raise HTTPException(status_code=404, detail="Benchmark Lab is disabled.")
    if not developer_token_configured():
        raise HTTPException(status_code=503, detail="Developer access is locked because HUMANIZER_DEVELOPER_TOKEN is not configured.")
    if not verify_developer_token(token):
        raise HTTPException(status_code=403, detail="Invalid developer password.")


@app.get("/api/developer/benchmark/status")
def benchmark_status_endpoint(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    return {"benchmark": corpus_status(), "calibration": calibration_status(), "validation": validation_status(), "reference_lm": reference_lm_status(), "registry": registry_status()}


@app.post("/api/developer/benchmark/sample")
def benchmark_add_sample(payload: BenchmarkSampleRequest, x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    if payload.provenance not in ALLOWED_PROVENANCE:
        raise HTTPException(status_code=400, detail=f"Unsupported provenance. Choose one of: {', '.join(sorted(ALLOWED_PROVENANCE))}")
    try:
        sample = add_sample(
            text=_validate_size(payload.text),
            provenance=payload.provenance,
            source_family=payload.source_family,
            discipline=payload.discipline,
            document_type=payload.document_type,
            editing_level=payload.editing_level,
            external_scores=payload.external_scores,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "sample": {k: v for k, v in sample.items() if k != "text"}, "benchmark": corpus_status()}


@app.post("/api/developer/benchmark/train")
def benchmark_train(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    try:
        result = train_from_benchmark()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "calibration": calibration_status(), "validation": result["validation"], "selected_model": result["model"].get("model_type"), "candidate_metrics": result["model"].get("candidate_metrics") or {}, "registry": result.get("registry") or registry_status()}


@app.post("/api/developer/benchmark/evaluate")
def benchmark_evaluate(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    try:
        report = evaluate_model()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "validation": report}


@app.get("/api/developer/benchmark/registry")
def benchmark_registry(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    return {"ok": True, "registry": registry_status()}


@app.post("/api/developer/benchmark/promote")
def benchmark_promote(payload: ModelRegistryRequest, x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    try:
        result = promote_model(payload.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "result": result, "registry": registry_status(), "calibration": calibration_status()}


@app.post("/api/developer/benchmark/rollback")
def benchmark_rollback(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    try:
        result = rollback_model()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "result": result, "registry": registry_status(), "calibration": calibration_status()}


@app.get("/api/developer/benchmark/drift")
def benchmark_drift(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    return {"ok": True, "drift": drift_status()}


@app.post("/api/developer/benchmark/adversarial-audit")
def adversarial_audit(payload: AdversarialAuditRequest, x_developer_token: str | None = Header(default=None)) -> dict:
    _require_benchmark_access(x_developer_token)
    original = _validate_size(payload.text)
    before = dashboard_report(original, include_private=True)
    engine1_text, engine1 = humanize_scholarly_text(original, mode=payload.mode)
    engine1_after = dashboard_report(engine1_text, include_private=True)
    engine3_text, engine3 = humanize_signal_guided(original, detector=before.get("ai_detector", {}), mode=payload.mode)
    engine3_after = dashboard_report(engine3_text, include_private=True)
    return {
        "ok": True,
        "before": before.get("ai_detection_percentage", 0),
        "engine1": {
            "changed": engine1_text != original,
            "after": engine1_after.get("ai_detection_percentage", 0),
            "preservation": preservation_certificate(original, engine1_text),
            "report": engine1,
        },
        "engine3": {
            "changed": engine3_text != original,
            "after": engine3_after.get("ai_detection_percentage", 0),
            "preservation": preservation_certificate(original, engine3_text),
            "report": engine3,
        },
        "note": "Robustness audit only. The purpose is to measure detector stability after editing, not to promise evasion of third-party detectors.",
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    content = await _read_upload_limited(file)
    filename = file.filename or "upload.txt"
    suffix = Path(filename).suffix.lower()
    document_id: str | None = None
    format_info: dict | None = None
    try:
        if suffix == ".docx" and _DOCX_FORMAT_PRESERVATION:
            structure = inspect_docx(content)
            text = str(structure["text"])
            document_id = uuid.uuid4().hex
            now = time.time()
            with _DOCUMENT_STORE_LOCK:
                _DOCUMENT_STORE[document_id] = {
                    "document_id": document_id,
                    "filename": filename,
                    "content": content,
                    "structure": structure,
                    "text": text,
                    "source_digest": structure.get("source_digest") or text_digest(text),
                    "humanized_outputs": {},
                    "created_at": now,
                    "updated_at": now,
                }
            format_info = {
                "available": True,
                "mode": "patch_original_docx",
                "paragraphs": int(structure.get("paragraph_count") or 0),
                "editable_paragraphs": int(structure.get("editable_paragraphs") or 0),
                "locked_paragraphs": int(structure.get("locked_paragraphs") or 0),
                "tables": int(structure.get("table_count") or 0),
                "locked_reason_counts": structure.get("locked_reason_counts") or {},
            }
        else:
            text = extract_text(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    text = _validate_size(text)
    return {
        "filename": filename,
        "text": text,
        "characters": len(text),
        "words": len(text.split()),
        "document_id": document_id,
        "format_preservation": format_info or {"available": False, "mode": "text_export"},
    }


@app.post("/api/analyse")
def analyse(payload: TextRequest) -> dict:
    text = _validate_size(payload.text)
    return dashboard_report(text)


def _humanize_core(payload: HumanizeRequest, progress: Callable[[int, str], None] | None = None, checkpoint: Callable[[dict], None] | None = None) -> dict:
    progress = progress or (lambda _progress, _stage: None)
    checkpoint = checkpoint or (lambda _state: None)
    original = _validate_size(payload.text)
    selected_engine = "engine2" if payload.use_model else payload.engine
    actual_engine = selected_engine
    progress(8, "Analysing source text…")
    original_dashboard = dashboard_report(original)
    progress(18, "Preparing rewrite engine…")

    engine1_report = {
        "applied": False, "engine": "engine1", "label": "Engine 1, Local rewrite",
        "reason": "Engine 1 was not selected.",
    }
    engine2_report = {
        "applied": False, "engine": "engine2", "label": "Engine 2, API rewrite",
        "reason": "Engine 2 was not selected.",
    }
    engine3_report = {
        "applied": False, "engine": "engine3", "label": "Engine 3, Signal-Guided rewrite",
        "reason": "Engine 3 was not selected.",
    }
    revised = original

    if selected_engine == "engine1":
        progress(28, "Running Engine 1 local rewrite…")
        revised, engine1_report = humanize_scholarly_text(original, mode=payload.mode)

    elif selected_engine == "engine3":
        progress(28, "Running Engine 3 signal-guided rewrite…")
        revised, engine3_report = humanize_signal_guided(
            original,
            detector=original_dashboard.get("ai_detector", {}),
            mode=payload.mode,
            segments=original_dashboard.get("segments", []),
        )

    elif selected_engine == "engine2":
        progress(24, "Running protected local preparation…")
        # Engine 2 begins from the independent local cleanup so the API spends its
        # effort on higher-level prose rather than obvious mechanical artifacts.
        engine1_text, engine1_report = humanize_scholarly_text(original, mode=payload.mode)
        status = provider_status()
        if not status.get("configured"):
            actual_engine = "engine1_fallback"
            revised = engine1_text
            engine2_report = {
                "applied": False, "engine": "engine2", "label": "Engine 2, API rewrite",
                "reason": "Engine 2 is not available on this deployment. Contact the administrator.",
                "fallback_used": True,
            }
        elif payload.mode in {"balanced", "deep"}:
            progress(38, "Engine 2 API refinement in progress…")
            revised, engine2_report = refine_with_model(
                engine1_text,
                mode=payload.mode,
                model_override=payload.engine2_model,
                progress_callback=lambda pct, stage: progress(38 + int(max(0, min(100, pct)) * 0.34), stage),
                checkpoint_callback=checkpoint,
            )
        else:
            actual_engine = "engine1_fallback"
            revised = engine1_text
            engine2_report = {
                "applied": False, "engine": "engine2", "label": "Engine 2, API rewrite",
                "reason": "Engine 2 is available only in balanced or deep mode. Engine 1 fallback was used.",
                "fallback_used": True,
            }

    progress(74, "Checking rewrite quality and preservation…")
    revised_dashboard = dashboard_report(revised)
    progress(88, "Running independent post-rewrite audit…")
    before_naturalness = int(original_dashboard.get("naturalness_percentage", 0))
    after_naturalness = int(revised_dashboard.get("naturalness_percentage", 0))
    before_ai_signal = int(original_dashboard.get("ai_detection_percentage", 0))
    after_ai_signal = int(revised_dashboard.get("ai_detection_percentage", 0))
    before_human_like = 100 - before_ai_signal
    after_human_like = 100 - after_ai_signal

    # Writing-quality guard applies to all engines. Engine 3 is intentionally
    # detector-coupled, but it still cannot trade scholarly quality for a lower score.
    if after_naturalness < before_naturalness:
        revised = original
        revised_dashboard = original_dashboard
        after_naturalness = before_naturalness
        after_ai_signal = before_ai_signal
        after_human_like = before_human_like
        if selected_engine == "engine2":
            engine2_report = {**engine2_report, "applied": False, "reason": "Final API rewrite was rejected because it degraded the protected rewrite-quality profile."}
            actual_engine = "engine1_fallback" if engine1_report.get("applied") else "none"
        elif selected_engine == "engine3":
            engine3_report = {**engine3_report, "applied": False, "reason": "Signal-guided candidate was rejected because it degraded the protected rewrite-quality profile."}
            actual_engine = "none"
        else:
            engine1_report = {**engine1_report, "applied": False, "reason": "Local candidate was rejected because it degraded the protected rewrite-quality profile."}
            actual_engine = "none"

    before_signals = {str(x.get("key")): int(x.get("score") or 0) for x in original_dashboard.get("ai_detector", {}).get("signals", [])}
    after_signals = {str(x.get("key")): int(x.get("score") or 0) for x in revised_dashboard.get("ai_detector", {}).get("signals", [])}
    targeted = list(engine3_report.get("targeted_signals") or []) if selected_engine == "engine3" else []
    targeted_before = sum(before_signals.get(key, 0) for key in targeted)
    targeted_after = sum(after_signals.get(key, 0) for key in targeted)
    if selected_engine == "engine3":
        signal_map = []
        for key in "ABCDEFGHI":
            before_value = before_signals.get(key, 0)
            after_value = after_signals.get(key, 0)
            signal_map.append({
                "key": key, "before": before_value, "after": after_value,
                "change": after_value - before_value, "targeted": key in targeted,
            })
        before_stats = original_dashboard.get("statistical_fingerprint_components", {}) or {}
        after_stats = revised_dashboard.get("statistical_fingerprint_components", {}) or {}
        statistical_map = [
            {"key": key, "before": int(before_stats.get(key, 0) or 0), "after": int(after_stats.get(key, 0) or 0), "change": int(after_stats.get(key, 0) or 0) - int(before_stats.get(key, 0) or 0)}
            for key in sorted(set(before_stats) | set(after_stats))
        ]
        engine3_report = {
            **engine3_report,
            "targeted_score_before": targeted_before,
            "targeted_score_after": targeted_after,
            "targeted_score_reduction": targeted_before - targeted_after,
            "signal_map": signal_map,
            "statistical_map": statistical_map,
            "section_profile_before": original_dashboard.get("section_profile", {}),
            "section_profile_after": revised_dashboard.get("section_profile", {}),
        }

    preservation = preservation_certificate(original, revised)
    progress(96, "Preparing revised text…")

    return {
        "selected_engine": selected_engine,
        "actual_engine": actual_engine,
        "selected_engine2_model": ({"gpt-5.6-luna": "v1", "gpt-5.6-terra": "v2"}.get(payload.engine2_model, payload.engine2_model)) if selected_engine == "engine2" else None,
        "changed": revised != original,
        "preservation_certificate": preservation,
        "original_report": original_dashboard,
        "text": revised,
        "report": revised_dashboard,
        "naturalness_improvement": {"before": before_naturalness, "after": after_naturalness, "gain": after_naturalness - before_naturalness},
        "ai_signal_improvement": {"before": before_ai_signal, "after": after_ai_signal, "reduction": before_ai_signal - after_ai_signal},
        "human_like_style_improvement": {"before": before_human_like, "after": after_human_like, "gain": after_human_like - before_human_like},
        "engine_1": engine1_report,
        "engine_2": engine2_report,
        "engine_3": engine3_report,
        "local_humanizer": engine1_report,
        "model_refiner": engine2_report,
    }



def _structured_paragraph_items(structure: dict) -> list[dict]:
    editable = [dict(item) for item in structure.get("paragraphs", []) if item.get("editable") and int(item.get("word_count") or 0) >= 6]
    for position, item in enumerate(editable):
        before = str(editable[position - 1].get("text") or "") if position > 0 else ""
        after = str(editable[position + 1].get("text") or "") if position + 1 < len(editable) else ""
        item["context_before"] = before[-1200:]
        item["context_after"] = after[:1200]
    return editable


def _merge_document_preservation(text_certificate: dict, word_certificate: dict) -> dict:
    checks = dict(text_certificate.get("checks") or {})
    checks.update(word_certificate.get("checks") or {})
    return {
        "passed": bool(text_certificate.get("passed")) and bool(word_certificate.get("passed")),
        "checks": checks,
        "changed_word_ratio": text_certificate.get("changed_word_ratio", 0),
        "changed_paragraphs": word_certificate.get("changed_paragraphs", 0),
        "locked_paragraphs": word_certificate.get("locked_paragraphs", 0),
        "note": "Protected scholarly evidence passed together with the Word-format preservation audit. The exported DOCX is patched into the original Word package rather than rebuilt from extracted text.",
    }


def _humanize_structured_docx(
    payload: HumanizeRequest,
    document: dict,
    *,
    job_id: str,
    progress: Callable[[int, str], None] | None = None,
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """Humanize only safe top-level Word paragraphs and patch the original DOCX.

    Tables, captions, headings, references, fields, hyperlinks, equations, mixed
    inline formatting, headers, footers, section settings and media stay in the
    original OOXML package. This prevents the format damage caused by the former
    extract-text → rebuild-DOCX workflow.
    """
    progress = progress or (lambda _pct, _stage: None)
    checkpoint = checkpoint or (lambda _state: None)
    structure = document["structure"]
    original = str(document["text"])
    submitted = _validate_size(payload.text)
    if text_digest(submitted) != str(document.get("source_digest") or text_digest(original)):
        raise HTTPException(
            status_code=409,
            detail="The Source text was edited after the DOCX was uploaded. Re-upload the Word file before humanizing if you want format-preserving Word export.",
        )

    selected_engine = "engine2" if payload.use_model else payload.engine
    actual_engine = selected_engine
    progress(7, "Reading original Word structure…")
    original_dashboard = dashboard_report(original)
    items = _structured_paragraph_items(structure)
    originals = {int(item["paragraph_index"]): str(item.get("text") or "") for item in items}
    replacements = dict(originals)
    total = max(1, len(items))
    progress(16, f"Locked Word structure preserved. {len(items)} prose paragraph(s) are eligible for editing…")

    engine1_report = {"applied": False, "engine": "engine1", "label": "Engine 1, Local rewrite", "reason": "Engine 1 was not selected.", "format_preserving_docx": True}
    engine2_report = {"applied": False, "engine": "engine2", "label": "Engine 2, API rewrite", "reason": "Engine 2 was not selected.", "format_preserving_docx": True}
    engine3_report = {"applied": False, "engine": "engine3", "label": "Engine 3, Signal-Guided rewrite", "reason": "Engine 3 was not selected.", "format_preserving_docx": True}

    if selected_engine == "engine1":
        changed = 0
        for pos, item in enumerate(items, start=1):
            pid = int(item["paragraph_index"])
            source_text = originals[pid]
            candidate, _report = humanize_scholarly_text(source_text, mode=payload.mode)
            local_limit = 0.60 if len(source_text.split()) < 120 else max(0.14, min(0.48, 90 / max(1, len(source_text.split()))))
            valid, _issues = validate_humanizer_preservation(source_text, candidate, max_word_change_ratio=local_limit)
            if valid:
                replacements[pid] = candidate
                changed += int(candidate != source_text)
            if pos == total or pos % max(1, total // 12) == 0:
                progress(18 + int((pos / total) * 50), f"Engine 1: {pos}/{total} Word paragraphs processed…")
        engine1_report = {
            "applied": changed > 0, "engine": "engine1", "label": "Engine 1, Local rewrite",
            "format_preserving_docx": True, "eligible_paragraphs": len(items), "changed_paragraphs": changed,
            "locked_paragraphs": int(structure.get("locked_paragraphs") or 0),
        }

    elif selected_engine == "engine3":
        changed = 0
        targeted: set[str] = set()
        for pos, item in enumerate(items, start=1):
            pid = int(item["paragraph_index"])
            source_text = originals[pid]
            local_dashboard = dashboard_report(source_text)
            candidate, local_report = humanize_signal_guided(
                source_text,
                detector=local_dashboard.get("ai_detector", {}),
                mode=payload.mode,
                segments=local_dashboard.get("segments", []),
            )
            local_limit = 0.60 if len(source_text.split()) < 120 else max(0.14, min(0.48, 90 / max(1, len(source_text.split()))))
            valid, _issues = validate_humanizer_preservation(source_text, candidate, max_word_change_ratio=local_limit)
            if valid:
                replacements[pid] = candidate
                changed += int(candidate != source_text)
            targeted.update(local_report.get("targeted_signals") or [])
            if pos == total or pos % max(1, total // 12) == 0:
                progress(18 + int((pos / total) * 50), f"Engine 3: {pos}/{total} eligible Word paragraphs screened and rewritten where signalled…")
        engine3_report = {
            "applied": changed > 0, "engine": "engine3", "label": "Engine 3, Signal-Guided rewrite",
            "format_preserving_docx": True, "eligible_paragraphs": len(items), "changed_paragraphs": changed,
            "locked_paragraphs": int(structure.get("locked_paragraphs") or 0), "targeted_signals": sorted(targeted),
        }

    elif selected_engine == "engine2":
        progress(20, "Preparing safe Word paragraphs for Engine 2…")
        seed_items: list[dict] = []
        local_changed = 0
        for item in items:
            pid = int(item["paragraph_index"])
            source_text = originals[pid]
            local_candidate, _local_report = humanize_scholarly_text(source_text, mode=payload.mode)
            local_limit = 0.60 if len(source_text.split()) < 120 else max(0.14, min(0.48, 90 / max(1, len(source_text.split()))))
            valid, _issues = validate_humanizer_preservation(source_text, local_candidate, max_word_change_ratio=local_limit)
            text_for_api = local_candidate if valid else source_text
            local_changed += int(text_for_api != source_text)
            seed_items.append({**item, "text": text_for_api})
        engine1_report = {
            "applied": local_changed > 0, "engine": "engine1", "label": "Engine 1, Local preparation",
            "format_preserving_docx": True, "eligible_paragraphs": len(items), "changed_paragraphs": local_changed,
        }
        status = provider_status()
        if not status.get("configured") or payload.mode == "light":
            actual_engine = "engine1_fallback"
            replacements = {int(item["paragraph_index"]): str(item["text"]) for item in seed_items}
            engine2_report = {
                "applied": False, "engine": "engine2", "label": "Engine 2, API rewrite", "format_preserving_docx": True,
                "reason": "Engine 2 is not available for this request. The protected local preparation was retained.", "fallback_used": True,
            }
        else:
            replacements, engine2_report = refine_paragraphs_with_model(
                seed_items,
                mode=payload.mode,
                model_override=payload.engine2_model,
                progress_callback=lambda pct, stage: progress(24 + int(max(0, min(100, pct)) * 0.48), stage),
                checkpoint_callback=checkpoint,
            )

    progress(75, "Reassembling revised prose into the original Word structure…")
    revised = render_structured_text(structure, replacements)
    revised_dashboard = dashboard_report(revised)
    before_naturalness = int(original_dashboard.get("naturalness_percentage", 0))
    after_naturalness = int(revised_dashboard.get("naturalness_percentage", 0))
    before_ai_signal = int(original_dashboard.get("ai_detection_percentage", 0))
    after_ai_signal = int(revised_dashboard.get("ai_detection_percentage", 0))

    # Keep the original Word package if the final document-level writing-quality
    # check worsens. Paragraph-level preservation already guards factual content.
    if after_naturalness < before_naturalness:
        replacements = dict(originals)
        revised = original
        revised_dashboard = original_dashboard
        after_naturalness = before_naturalness
        after_ai_signal = before_ai_signal
        actual_engine = "none"
        if selected_engine == "engine1":
            engine1_report = {**engine1_report, "applied": False, "reason": "Rewrites were discarded because the document-level writing-quality profile worsened."}
        elif selected_engine == "engine2":
            engine2_report = {**engine2_report, "applied": False, "reason": "Rewrites were discarded because the document-level writing-quality profile worsened."}
        else:
            engine3_report = {**engine3_report, "applied": False, "reason": "Rewrites were discarded because the document-level writing-quality profile worsened."}

    progress(84, "Patching original DOCX paragraphs without rebuilding tables or layout…")
    changed_replacements = {pid: value for pid, value in replacements.items() if value != originals.get(pid, value)}
    patched = patch_docx(document["content"], structure, changed_replacements)
    word_certificate = format_preservation_certificate(
        document["content"], patched,
        changed_paragraphs=len(changed_replacements),
        locked_paragraphs=int(structure.get("locked_paragraphs") or 0),
    )
    text_certificate = preservation_certificate(original, revised)
    combined_certificate = _merge_document_preservation(text_certificate, word_certificate)
    if not combined_certificate["passed"]:
        raise RuntimeError("Word structure preservation audit failed. Export was blocked and the original DOCX was left unchanged.")

    progress(91, "Running independent post-rewrite audit…")
    _store_humanized_document(str(document["document_id"]), job_id, patched, combined_certificate)
    after_human_like = 100 - after_ai_signal
    before_human_like = 100 - before_ai_signal
    progress(97, "Format-preserving Word document is ready…")

    return {
        "selected_engine": selected_engine,
        "actual_engine": actual_engine,
        "selected_engine2_model": ({"gpt-5.6-luna": "v1", "gpt-5.6-terra": "v2"}.get(payload.engine2_model, payload.engine2_model)) if selected_engine == "engine2" else None,
        "changed": revised != original,
        "text": revised,
        "report": revised_dashboard,
        "original_report": original_dashboard,
        "preservation_certificate": combined_certificate,
        "format_preservation_certificate": word_certificate,
        "format_preserving_export": True,
        "document_id": str(document["document_id"]),
        "humanize_job_id": job_id,
        "document_structure": {
            "paragraphs": int(structure.get("paragraph_count") or 0),
            "tables": int(structure.get("table_count") or 0),
            "editable_paragraphs": int(structure.get("editable_paragraphs") or 0),
            "locked_paragraphs": int(structure.get("locked_paragraphs") or 0),
            "changed_paragraphs": len(changed_replacements),
        },
        "naturalness_improvement": {"before": before_naturalness, "after": after_naturalness, "gain": after_naturalness - before_naturalness},
        "ai_signal_improvement": {"before": before_ai_signal, "after": after_ai_signal, "reduction": before_ai_signal - after_ai_signal},
        "human_like_style_improvement": {"before": before_human_like, "after": after_human_like, "gain": after_human_like - before_human_like},
        "engine_1": engine1_report,
        "engine_2": engine2_report,
        "engine_3": engine3_report,
        "local_humanizer": engine1_report,
        "model_refiner": engine2_report,
    }

@app.post("/api/humanize")
def humanize(payload: HumanizeRequest) -> dict:
    """Backward-compatible synchronous endpoint. The v2.4 UI uses jobs below."""
    document = _get_document(payload.document_id)
    if payload.document_id and document is None:
        raise HTTPException(status_code=404, detail="The uploaded Word document has expired. Re-upload it before format-preserving humanization.")
    if document is not None:
        sync_id = "sync-" + uuid.uuid4().hex
        return _humanize_structured_docx(payload, document, job_id=sync_id)
    return _humanize_core(payload)

def _run_humanize_job(job_id: str, payload_data: dict) -> None:
    try:
        _update_humanize_job(job_id, status="running", progress=5, stage="Starting humanization…")
        payload = HumanizeRequest(**payload_data)
        document = _get_document(payload.document_id)
        if payload.document_id and document is None:
            raise RuntimeError("The uploaded Word document has expired. Re-upload it before format-preserving humanization.")
        if document is not None:
            result = _humanize_structured_docx(
                payload, document, job_id=job_id,
                progress=lambda pct, stage: _job_progress(job_id, pct, stage),
                checkpoint=lambda state: _job_checkpoint(job_id, state),
            )
        else:
            result = _humanize_core(
                payload,
                progress=lambda pct, stage: _job_progress(job_id, pct, stage),
                checkpoint=lambda state: _job_checkpoint(job_id, state),
            )
        _update_humanize_job(
            job_id,
            status="completed",
            progress=100,
            stage="Humanization completed",
            result=result,
            error=None,
        )
    except Exception as exc:
        # Keep the browser connection healthy and surface a useful error through polling.
        _update_humanize_job(
            job_id,
            status="failed",
            progress=100,
            stage="Humanization failed",
            error=str(exc) or exc.__class__.__name__,
        )

@app.post("/api/humanize/jobs", status_code=202)
def start_humanize_job(payload: HumanizeRequest) -> dict:
    _clean_humanize_jobs()
    # Validate the large field before returning 202 so obviously invalid jobs fail immediately.
    _validate_size(payload.text)
    job_id = uuid.uuid4().hex
    now = time.time()
    with _HUMANIZE_JOBS_LOCK:
        _HUMANIZE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 2,
            "stage": "Queued for humanization…",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "checkpoint": None,
            "checkpoint_summary": None,
        }
    _HUMANIZE_JOB_EXECUTOR.submit(_run_humanize_job, job_id, payload.model_dump() if hasattr(payload, "model_dump") else payload.dict())
    return {"job_id": job_id, "status": "queued", "progress": 2, "stage": "Queued for humanization…"}

@app.get("/api/humanize/jobs/{job_id}")
def humanize_job_status(job_id: str) -> dict:
    _clean_humanize_jobs()
    with _HUMANIZE_JOBS_LOCK:
        job = _HUMANIZE_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Humanization job was not found or has expired.")
        # Return a copy so the worker can update safely after this response is built.
        # Do not send the partial manuscript on every poll; only compact progress
        # metadata is public to the active browser job.
        response = dict(job)
        response.pop("checkpoint", None)
        return response


@app.post("/api/export/docx")
def export_docx(payload: ExportRequest) -> Response:
    text = _validate_size(payload.text)
    filename = "scholarly_humanized_text.docx"
    if not payload.annotated and payload.document_id and payload.humanize_job_id:
        document = _get_document(payload.document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="The original Word upload has expired. Re-upload and humanize it again for format-preserving export.")
        output = (document.get("humanized_outputs") or {}).get(payload.humanize_job_id)
        if output is None:
            raise HTTPException(status_code=404, detail="The format-preserving Word output for this humanization job is no longer available.")
        certificate = output.get("certificate") or {}
        if not certificate.get("passed"):
            raise HTTPException(status_code=409, detail="Word format preservation audit did not pass, so export was blocked.")
        content = bytes(output["content"])
        stem = Path(str(document.get("filename") or "manuscript.docx")).stem
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "manuscript"
        filename = f"{safe_stem}_humanized.docx"
    elif payload.annotated:
        report = dashboard_report(text)
        content = build_annotated_docx(text, report["segments"], payload.title)
        filename = "ai_signal_diagnostic.docx"
    else:
        content = build_docx(text, payload.title)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/export/html")
def export_html(payload: ExportRequest) -> Response:
    text = _validate_size(payload.text)
    report = dashboard_report(text)
    body = report["signal_coloured_html"] if payload.annotated else html.escape(text).replace("\n", "<br>")
    legend = (
        "<div class='legend'><b>A–I signal colours:</b> A predictability · B burstiness · C hedging · D structure · "
        "E specificity · F transitions · G punctuation · H voice/register · I rhetorical scaffolding.</div>"
        if payload.annotated
        else ""
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(payload.title)}</title>
    <style>body{{font-family:Arial,sans-serif;max-width:980px;margin:40px auto;line-height:1.75;padding:0 24px}}
    .legend{{padding:12px;background:#f8fafc;border:1px solid #e2e8f0;margin-bottom:20px}}
    .signal-text{{text-decoration-line:underline;text-decoration-thickness:3px;text-underline-offset:4px}}
    .signal-badge{{display:inline-block;margin-left:3px;padding:1px 4px;border-radius:999px;font:700 9px Arial}}
    .signal-A,.signal-text-A{{color:#6b21a8;text-decoration-color:#9333ea}} .signal-B,.signal-text-B{{color:#1d4ed8;text-decoration-color:#2563eb}}
    .signal-C,.signal-text-C{{color:#0f766e;text-decoration-color:#0d9488}} .signal-D,.signal-text-D{{color:#c2410c;text-decoration-color:#ea580c}}
    .signal-E,.signal-text-E{{color:#15803d;text-decoration-color:#16a34a}} .signal-F,.signal-text-F{{color:#0e7490;text-decoration-color:#0891b2}}
    .signal-G,.signal-text-G{{color:#be123c;text-decoration-color:#e11d48}} .signal-H,.signal-text-H{{color:#4338ca;text-decoration-color:#4f46e5}}
    .signal-I,.signal-text-I{{color:#be185d;text-decoration-color:#db2777}} .protected-text{{border-bottom:1px dotted #94a3b8}}
    </style>
    </head><body><h1>{html.escape(payload.title)}</h1>{legend}<div>{body}</div></body></html>"""
    return Response(
        content=document.encode("utf-8"),
        media_type="text/html",
        headers={"Content-Disposition": 'attachment; filename="scholarly_humanizer.html"'},
    )
