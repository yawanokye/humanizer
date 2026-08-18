from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scholarly_humanizer import humanize_scholarly_text, humanize_signal_guided
from services.analyzer import dashboard_report
from services.document_io import build_annotated_docx, build_docx, extract_text
from services.model_refiner import provider_status, refine_with_model

BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = int(os.getenv("HUMANIZER_MAX_INPUT_CHARS", "1200000"))
MAX_UPLOAD_BYTES = int(os.getenv("HUMANIZER_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Scholarly Humanizer",
    version="2.1.0",
    description="Four-layer AI-style screening with signal-coloured diagnostics, independent Engine 1, optional API Engine 2, and preservation-gated signal-guided Engine 3.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class HumanizeRequest(TextRequest):
    mode: Literal["light", "balanced", "deep"] = "balanced"
    engine: Literal["engine1", "engine2", "engine3"] = "engine1"
    engine2_model: Literal["gpt-5.6-terra", "gpt-5.6-luna"] = "gpt-5.6-terra"
    # Backward-compatible field for older frontends. New UI uses engine.
    use_model: bool = False


class ExportRequest(TextRequest):
    title: str = "Scholarly Humanized Text"
    annotated: bool = False


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
    provider = provider_status()
    engines = dict(provider.get("engines", {}) or {})
    engines.setdefault("engine1", {"configured": True, "label": "Engine 1, Local rewrite"})
    engines["engine3"] = {
        "configured": True,
        "label": "Engine 3, Signal-Guided rewrite",
        "message": "Local signal-guided rewrite is ready. It targets detected A-I style signals and uses the preservation gate.",
    }
    return {
        "ok": True,
        "version": app.version,
        "model": provider,
        "engines": engines,
        "max_input_chars": MAX_INPUT_CHARS,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    content = await _read_upload_limited(file)
    try:
        text = extract_text(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _validate_size(text)
    return {"filename": file.filename, "text": text, "report": dashboard_report(text)}


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
                "reason": status.get("engines", {}).get("engine2", {}).get("message", "Engine 2 API rewrite is not configured."),
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
        engine3_report = {
            **engine3_report,
            "targeted_score_before": targeted_before,
            "targeted_score_after": targeted_after,
            "targeted_score_reduction": targeted_before - targeted_after,
        }

    return {
        "selected_engine": selected_engine,
        "actual_engine": actual_engine,
        "selected_engine2_model": payload.engine2_model if selected_engine == "engine2" else None,
        "changed": revised != original,
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
