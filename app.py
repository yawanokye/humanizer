from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scholarly_humanizer import humanize_scholarly_text, humanize_signal_guided, preservation_certificate
from services.analyzer import dashboard_report
from services.calibration import calibration_status
from services.reference_lm import reference_lm_status
from services.benchmark import (
    ALLOWED_PROVENANCE, add_sample, benchmark_enabled, corpus_status, developer_token_configured, evaluate_model,
    train_from_benchmark, validation_status, verify_developer_token, registry_status,
    promote_model, rollback_model, drift_status,
)
from services.document_io import build_annotated_docx, build_docx, extract_text
from services.model_refiner import provider_status, refine_with_model

BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = int(os.getenv("HUMANIZER_MAX_INPUT_CHARS", "1200000"))
MAX_UPLOAD_BYTES = int(os.getenv("HUMANIZER_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Scholarly Humanizer",
    version="2.4.0",
    description="Private-calibrated, robustness-tested scholarly AI-style screening with locked-test validation, model registry, signal-coloured diagnostics, independent Engine 1, API Engine 2, and signal-guided Engine 3.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class HumanizeRequest(TextRequest):
    mode: Literal["light", "balanced", "deep"] = "balanced"
    engine: Literal["engine1", "engine2", "engine3"] = "engine1"
    engine2_model: Literal["v1", "v2", "gpt-5.6-terra", "gpt-5.6-luna"] = "v2"
    # Backward-compatible field for older frontends. New UI uses engine.
    use_model: bool = False


class ExportRequest(TextRequest):
    title: str = "Scholarly Humanized Text"
    annotated: bool = False


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
    try:
        text = extract_text(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    text = _validate_size(text)
    return {
        "filename": file.filename or "upload",
        "text": text,
        "characters": len(text),
        "words": len(text.split()),
    }


@app.post("/api/analyse")
def analyse(payload: TextRequest) -> dict:
    text = _validate_size(payload.text)
    return dashboard_report(text)


@app.post("/api/humanize")
def humanize(payload: HumanizeRequest) -> dict:
    original = _validate_size(payload.text)
    selected_engine = "engine2" if payload.use_model else payload.engine
    actual_engine = selected_engine
    original_dashboard = dashboard_report(original)

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
        revised, engine1_report = humanize_scholarly_text(original, mode=payload.mode)

    elif selected_engine == "engine3":
        revised, engine3_report = humanize_signal_guided(
            original,
            detector=original_dashboard.get("ai_detector", {}),
            mode=payload.mode,
        )

    elif selected_engine == "engine2":
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
            revised, engine2_report = refine_with_model(engine1_text, mode=payload.mode, model_override=payload.engine2_model)
        else:
            actual_engine = "engine1_fallback"
            revised = engine1_text
            engine2_report = {
                "applied": False, "engine": "engine2", "label": "Engine 2, API rewrite",
                "reason": "Engine 2 is available only in balanced or deep mode. Engine 1 fallback was used.",
                "fallback_used": True,
            }

    revised_dashboard = dashboard_report(revised)
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


@app.post("/api/export/docx")
def export_docx(payload: ExportRequest) -> Response:
    text = _validate_size(payload.text)
    if payload.annotated:
        report = dashboard_report(text)
        content = build_annotated_docx(text, report["segments"], payload.title)
        filename = "ai_signal_diagnostic.docx"
    else:
        content = build_docx(text, payload.title)
        filename = "scholarly_humanized_text.docx"
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
