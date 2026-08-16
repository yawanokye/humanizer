# Scholarly Humanizer, Render-ready standalone app

A FastAPI web application built from the supplied `scholarly_humanizer.py` module. It improves scholarly naturalness without introducing deliberate mistakes, changing evidence, or claiming to detect authorship.

## Core features

- Paste text or upload TXT, Markdown, DOCX and text-based PDF files.
- Overall natural scholarly voice score and style-concern percentage.
- Category-level concern percentages for Primary Statistical Metrics, Linguistic and N-gram Patterns, Vocabulary and Stylistic Markers, and Semantic and Logic Constraints.
- Sentence-level colour map with explainable reasons.
- Engine 1, Local rewrite, protected humanisation that preserves headings, numbers, years, citations, URLs, equations, tables and action placeholders.
- Engine 2, API rewrite, optional OpenAI-compatible or remote Ollama refinement with preservation validation and automatic fallback.
- Light, balanced and deep modes.
- Clean DOCX, annotated DOCX and coloured HTML export.
- Stateless processing. The app does not require a database or persistent disk.
- Render health check, dynamic port binding, bounded uploads and production security headers.

## Important interpretation

The dashboard measures formulaic, repetitive, overloaded, rhythmically uniform and locally checkable logic-risk patterns. The percentages are writing-quality signals, not AI-detection probabilities, and they do not establish who wrote the text. Token-probability and perplexity indicators are local proxies, not direct model log-probability measurements.

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

For Engine 2 using an OpenAI-compatible service, add these environment variables in Render:

```text
HUMANIZER_PROVIDER=openai_compatible
HUMANIZER_MODEL=your-model-name
HUMANIZER_BASE_URL=https://your-provider.example/v1
HUMANIZER_API_KEY=your-secret-key
```

Mark `HUMANIZER_API_KEY` as a secret. Do not commit it to the repository.

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
