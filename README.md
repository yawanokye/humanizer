# Scholarly Humanizer, Render-ready standalone app

A FastAPI web application built from the supplied `scholarly_humanizer.py` module. It combines an explainable nine-signal AI-style detector with protected scholarly rewriting. It does not treat the detector score as proof of authorship.

## AI Detector and naturalness workflow

The dashboard is AI-detector first. It screens nine corroborating AI-style signal families and reports an explainable score, verdict, confidence and estimated AI-edited fraction. Naturalness is shown separately as a writing-quality score rather than being treated as proof of authorship.

The existing humanizer is the naturalness-improvement engine. Engine 1 generates progressively stronger preservation-safe local candidates for Light, Balanced and Deep modes, scores them, and keeps only the best candidate that does not reduce naturalness. Engine 2 applies the same preservation and non-degradation principle to API-refined batches. The interface reports the naturalness score before and after rewriting.

## Core features

- Paste text or upload TXT, Markdown, DOCX and text-based PDF files.
- AI Detector as the primary dashboard, with a 0–27 nine-signal score, verdict, confidence and AI-edited fraction estimate.
- Separate Naturalness score, shown as a writing-quality measure rather than the inverse of the AI score.
- Nine signal families: perplexity/predictability, burstiness, hedge density, structural tells, specificity, transitions, punctuation, voice/register and rhetorical scaffolding.
- Sentence-level AI-signal colour map with explainable reasons and category evidence.
- Engine 1, Local rewrite, protected humanisation that preserves headings, numbers, years, citations, URLs, equations, tables and action placeholders.
- Engine 2, API rewrite, optional OpenAI-compatible or remote Ollama refinement with preservation validation and automatic fallback.
- Light, balanced and deep modes.
- Clean DOCX, annotated DOCX and coloured HTML export.
- Stateless processing. The app does not require a database or persistent disk.
- Render health check, dynamic port binding, bounded uploads and production security headers.

## Important interpretation

The AI Detector is deliberately sensitive to clusters of formulaic, repetitive, rhythmically uniform and rhetorically scaffolded patterns. Its percentage is an AI-style signal score, not a calibrated probability and not proof of authorship. Strong verdicts require corroboration across multiple signal families. Academic prose is calibrated so normal hedging, neutral register, lists and semicolon use do not become strong evidence by themselves.

## Rewrite engines

- **Engine 1, Local rewrite:** default deterministic editor. It needs no OpenAI key and does not send text to an external model. It removes low-risk formulaic phrasing, repeated connectors and some overloaded sentence structures while preserving academic evidence signatures.
- **Engine 2, API rewrite:** optional API pass. Use it only when `HUMANIZER_PROVIDER`, `HUMANIZER_MODEL`, `HUMANIZER_BASE_URL` and, where required, `HUMANIZER_API_KEY` are configured. The app applies preservation checks and falls back to Engine 1 output if the API changes protected content.

## Deploy to Render with Blueprint

1. Upload the complete project folder to a GitHub, GitLab or Bitbucket repository.
2. In Render, choose **New > Blueprint**.
3. Connect the repository containing `render.yaml`.
4. Review the proposed `scholarly-humanizer` web service and deploy it.
5. Render builds the Docker image and checks `/healthz` before routing traffic.

The included Blueprint uses the Frankfurt region and the Free instance type. For sustained use, long documents, or simultaneous users, change `plan: free` to a paid instance type in `render.yaml` or upgrade the service in Render.

## Manual Render setup

Use these settings when creating a Web Service without the Blueprint:

- **Language:** Docker
- **Dockerfile path:** `./Dockerfile`
- **Health check path:** `/healthz`
- **Region:** Frankfurt
- **Docker command:** leave blank, the Dockerfile command is already correct

The server binds to `0.0.0.0` and uses Render's `PORT` environment variable automatically.

## Engine 2 API rewrite on Render

The default deployment uses Engine 1 local rewrite:

```text
HUMANIZER_PROVIDER=none
```

For Engine 2 using the OpenAI API, add these environment variables in Render:

```text
HUMANIZER_PROVIDER=openai
OPENAI_API_KEY=sk-your-secret-key
OPENAI_MODEL=gpt-5.6-terra
OPENAI_BASE_URL=https://api.openai.com/v1
```

Mark `OPENAI_API_KEY` as a secret. Do not commit it to the repository. `OPENAI_MODEL` defaults to `gpt-5.6-terra` if omitted. Use `gpt-5.6-luna` when lower cost and higher-volume processing matter more than maximum rewrite quality. `OPENAI_BASE_URL` is optional because the app defaults to `https://api.openai.com/v1` when `HUMANIZER_PROVIDER=openai`. The official OpenAI path uses the Responses API.

For another OpenAI-compatible provider, the existing `HUMANIZER_MODEL`, `HUMANIZER_BASE_URL`, and `HUMANIZER_API_KEY` variables remain supported.

A locally running Ollama instance on your computer cannot be reached through `localhost` from Render. Ollama must be hosted as a separate reachable service, preferably on Render's private network or another secured server.

## Resource settings

Default limits can be changed in the Render Dashboard:

```text
HUMANIZER_MAX_INPUT_CHARS=1200000
HUMANIZER_MAX_UPLOAD_BYTES=15728640
WEB_CONCURRENCY=1
HUMANIZER_TIMEOUT_SECONDS=180
```

Keep `WEB_CONCURRENCY=1` on small instances because document extraction and analysis can consume significant memory. Scale the service vertically or horizontally for heavier institutional use.

## Local Docker test

```bash
docker build -t scholarly-humanizer .
docker run --rm -p 10000:10000 -e PORT=10000 scholarly-humanizer
```

Open `http://127.0.0.1:10000` and verify `http://127.0.0.1:10000/healthz`.

## Local Python test

```bash
python -m pip install -r requirements.txt
PORT=10000 ./start.sh
```

On Windows, `start.bat` remains available for local use.

## Tests

```bash
python -m unittest discover -s tests -v
```

### Browser cache after upgrading from the old AI-enabled control

If a browser shows `Cannot read properties of null (reading 'checked')`, it is loading an older cached `app.js` that still expects the removed `useModel` checkbox. Version 1.3.0 cache-busts static assets, disables browser caching for the app shell/static JavaScript, and includes a hidden compatibility control so older cached code cannot crash the page. Redeploy this build and refresh the page once.
