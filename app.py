from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scholarly_humanizer import humanize_scholarly_text
from services.analyzer import dashboard_report
from services.document_io import build_annotated_docx, build_docx, extract_text
from services.model_refiner import provider_status, refine_with_model

BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = int(os.getenv("HUMANIZER_MAX_INPUT_CHARS", "1200000"))
MAX_UPLOAD_BYTES = int(os.getenv("HUMANIZER_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Scholarly Humanizer",
    version="1.7.0",
    description="Nine-signal AI-style detection with complementary Human-like Style scoring, Engine 1 local rewrite and optional Engine 2 API rewrite.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class HumanizeRequest(TextRequest):
    mode: Literal["light", "balanced", "deep"] = "balanced"
    engine: Literal["engine1", "engine2"] = "engine1"
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
    return {
        "ok": True,
        "version": app.version,
        "model": provider,
        "engines": provider.get("engines", {}),
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

    engine1_text, engine1_report = humanize_scholarly_text(original, mode=payload.mode)
    engine2_report = {
        "applied": False,
        "engine": "engine2",
        "label": "Engine 2, API rewrite",
        "reason": "Engine 2 was not selected.",
    }
    revised = engine1_text

    if selected_engine == "engine2":
        status = provider_status()
        if not status.get("configured"):
            actual_engine = "engine1_fallback"
            revised = engine1_text
            engine2_report = {
                "applied": False,
                "engine": "engine2",
                "label": "Engine 2, API rewrite",
                "reason": status.get("engines", {}).get("engine2", {}).get("message", "Engine 2 API rewrite is not configured."),
                "fallback_used": True,
            }
        elif payload.mode in {"balanced", "deep"}:
            revised, engine2_report = refine_with_model(engine1_text, mode=payload.mode, model_override=payload.engine2_model)
        else:
            actual_engine = "engine1_fallback"
            revised = engine1_text
            engine2_report = {
                "applied": False,
                "engine": "engine2",
                "label": "Engine 2, API rewrite",
                "reason": "Engine 2 is available only in balanced or deep mode. Engine 1 fallback was used.",
                "fallback_used": True,
            }

    original_dashboard = dashboard_report(original)
    revised_dashboard = dashboard_report(revised)
    before_naturalness = int(original_dashboard.get("naturalness_percentage", 0))
    after_naturalness = int(revised_dashboard.get("naturalness_percentage", 0))
    before_ai_signal = int(original_dashboard.get("ai_detection_percentage", 0))
    after_ai_signal = int(revised_dashboard.get("ai_detection_percentage", 0))
    before_human_like = 100 - before_ai_signal
    after_human_like = 100 - after_ai_signal

    # Final guard: a humanisation request must not degrade either the internal
    # rewrite-quality metric or the public complementary style profile. Engine 1
    # and Engine 2 already apply preservation gates, but this protects the fully
    # assembled document as well.
    if after_naturalness < before_naturalness or after_ai_signal > before_ai_signal:
        revised = original
        revised_dashboard = original_dashboard
        after_naturalness = before_naturalness
        after_ai_signal = before_ai_signal
        before_human_like = 100 - before_ai_signal
        after_human_like = before_human_like
        if selected_engine == "engine2":
            engine2_report = {**engine2_report, "applied": False, "reason": "Final API rewrite was rejected because it degraded the protected rewrite-quality or Human-like Style profile."}
            actual_engine = "engine1_fallback" if engine1_text != original else "none"

    return {
        "selected_engine": selected_engine,
        "actual_engine": actual_engine,
        "selected_engine2_model": payload.engine2_model if selected_engine == "engine2" else None,
        "changed": revised != original,
        "original_report": original_dashboard,
        "text": revised,
        "report": revised_dashboard,
        "naturalness_improvement": {
            "before": before_naturalness,
            "after": after_naturalness,
            "gain": after_naturalness - before_naturalness,
        },
        "ai_signal_improvement": {
            "before": before_ai_signal,
            "after": after_ai_signal,
            "reduction": before_ai_signal - after_ai_signal,
        },
        "human_like_style_improvement": {
            "before": before_human_like,
            "after": after_human_like,
            "gain": after_human_like - before_human_like,
        },
        "engine_1": engine1_report,
        "engine_2": engine2_report,
        # Backward-compatible response keys.
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
    body = report["highlighted_html"] if payload.annotated else html.escape(text).replace("\n", "<br>")
    legend = (
        "<div class='legend'><b>AI signal key:</b> red = high signal, orange = moderate, "
        "yellow = low, green = natural.</div>"
        if payload.annotated
        else ""
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(payload.title)}</title>
    <style>body{{font-family:Arial,sans-serif;max-width:980px;margin:40px auto;line-height:1.65;padding:0 24px}}
    .risk-high{{background:#fecaca}}.risk-moderate{{background:#fed7aa}}.risk-low{{background:#fef3c7}}.risk-natural{{background:#dcfce7}}
    .risk-protected{{border-bottom:1px dotted #94a3b8}} .legend{{padding:12px;background:#f8fafc;border:1px solid #e2e8f0;margin-bottom:20px}}</style>
    </head><body><h1>{html.escape(payload.title)}</h1>{legend}<div>{body}</div></body></html>"""
    return Response(
        content=document.encode("utf-8"),
        media_type="text/html",
        headers={"Content-Disposition": 'attachment; filename="scholarly_humanizer.html"'},
    )
