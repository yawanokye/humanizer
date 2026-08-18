# Scholarly Humanizer, Render-ready standalone app

A FastAPI web application built from the supplied `scholarly_humanizer.py` module. It combines an explainable four-layer AI-style detector with protected scholarly rewriting and three rewrite engines. It does not treat the detector score as proof of authorship.

## AI Detector and Human-like Style workflow

The dashboard is AI-detector first. It screens nine corroborating AI-style signal families and reports an explainable AI Signal Index, descriptive signal level, confidence, 0–27 forensic category score, evidence counts and sentence map. The second headline metric is **Human-like Style**, defined exactly as `100 - AI Signal`. The interface does not estimate what percentage of the document was "written by AI" because detector scores are not reliable authorship fractions.

The existing humanizer still uses its internal Naturalness metric to choose the best preservation-safe rewrite. Engine 1 generates progressively stronger local candidates for Light, Balanced and Deep modes, then keeps the strongest candidate that does not reduce internal rewrite quality. Deep is the default. Engine 2 applies the same preservation gate to API-refined batches. After rewriting, the visible AI Signal is recalculated from the revised text and Human-like Style is displayed as its exact complement.

## Core features

- Paste text or upload TXT, Markdown, DOCX and text-based PDF files.
- AI Detector as the primary dashboard, with a 0–100 AI Signal Index, a 0–27 nine-signal forensic score, descriptive signal level and confidence.
- Complementary Human-like Style score, always `100 - AI Signal`.
- Internal Naturalness scoring remains part of rewrite selection but is not a headline detector metric.
- Nine signal families: perplexity/predictability, burstiness, hedge density, structural tells, specificity, transitions, punctuation, voice/register and rhetorical scaffolding.
- Sentence-level AI-signal colour map with explainable reasons and category evidence. The headline diagnostic counters show active signal categories and evidence items so they reconcile with the document-level index.
- Engine 1, Local rewrite, evidence-locked humanisation that preserves names, emails, headings, numbers, years, citations, references, URLs, DOIs, equations, tables, form rows and action placeholders.
- Engine 2, API rewrite, optional OpenAI-compatible or remote Ollama refinement with preservation validation and automatic fallback.
- Engine 3, Signal-Guided rewrite, local preservation-gated editing that reads the active A–I detector profile and directly targets safely editable signals.
- Light, balanced and deep modes.
- Clean DOCX, annotated DOCX and coloured HTML export.
- Stateless processing. The app does not require a database or persistent disk.
- Render health check, dynamic port binding, bounded uploads and production security headers.

## Important interpretation

The AI Detector is deliberately sensitive to clusters of formulaic, repetitive, rhythmically uniform and rhetorically scaffolded patterns. Its percentage is an AI-style signal index, not a calibrated probability, not the percentage of words written by AI, and not proof of authorship. Version 1.8 retains the nine-signal forensic core but replaces authorship-style verdicts with descriptive bands: Minimal, Low, Moderate, Elevated and Strong AI-style signal. The dashboard also warns that commercial detectors can disagree substantially on the same scholarly passage because their models, thresholds and training data differ. Academic prose is calibrated so technical terms such as “robust”, ordinary statistical “between X and Y” phrasing, citation semicolons, normal lists and neutral scholarly register do not become strong evidence by themselves.

## Rewrite engines

- **Engine 1, Local rewrite:** default deterministic editor. It needs no OpenAI key and does not send text to an external model. Deep mode is now aggressive on editable prose cadence while names, emails, numbers, percentages, citations, references, tables, headings, equations and other evidence-bearing spans are locked. The engine can remove formulaic phrasing, repeated connectors and safe overloaded sentence joins without changing the protected content.
- **Engine 2, API rewrite:** OpenAI-ready API pass. The Render blueprint sets `HUMANIZER_PROVIDER=openai`, `OPENAI_MODEL=gpt-5.6-terra`, and the API base URL. Add the secret `OPENAI_API_KEY` during deployment. If the key is missing, the app now states that Engine 1 fallback was used instead of reporting an unchanged Engine 2 rewrite. The app applies preservation checks and falls back to Engine 1 output if the API changes protected content or lowers internal rewrite quality.

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

## Engine 2 selection and model choice

Engine 2 is always selectable in the browser. The UI no longer disables it when the API key is missing or while the status request is loading. When Engine 2 is selected, a second selector appears for **GPT-5.6 Terra** or **GPT-5.6 Luna**. Terra is the default quality/cost balance; Luna is the lower-cost high-volume option. The chosen model is sent with each humanization request and does not require changing the Render environment variable between runs. If `OPENAI_API_KEY` is missing, the UI keeps Engine 2 selectable but explains that API rewriting cannot run until the secret is configured.

## Engine 2 API rewrite on Render

The included Render blueprint preconfigures Engine 2 for OpenAI. Engine 1 still works without an API call and remains available from the interface. For Engine 2, supply these environment variables in Render:

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

If a browser shows `Cannot read properties of null (reading 'checked')`, it is loading an older cached `app.js` that still expects the removed `useModel` checkbox. Version 2.1.0 cache-busts static assets, disables browser caching for the app shell/static JavaScript, and includes a hidden compatibility control so older cached code cannot crash the page. Redeploy this build and refresh the page once.


## v2.0 independent detector and paragraph-aware rewrite

- AI Signal is now a composite style index rather than a direct conversion of one coarse 0–27 total. The visible calculation combines 30% weighted A–I categories, 40% paragraph/section distribution, 20% countable lexical/rhythm evidence and 10% document-level regularity.
- Long extracted documents are segmented paragraph-by-paragraph even when DOCX/TXT/PDF extraction places each paragraph on its own line. This prevents a few concrete methods sections from diluting a more formulaic paragraph elsewhere.
- Humanness/context evidence is still reported, but it adjusts confidence only. It no longer subtracts AI-style evidence from the headline score.
- Engine 1 is detector-independent. It selects rewrites using preservation and writing-quality checks only. The AI detector runs afterwards as an independent audit, so Engine 1 cannot simply optimise against its own detector thresholds.
- Deep mode preserves paragraph boundaries and evidence-bearing content while making safe changes to wordy formulaic phrasing, repeated openings, mechanical transitions and overloaded sentence structure.
- The dashboard now shows prose segments screened, flagged prose segments, paragraph hotspots and the exact composite score components.
- Tables, form rows, headings, references, equations, emails, citations, numbers and table punctuation remain excluded or locked as appropriate.


## v2.1 statistical fingerprint, Engine 3 and signal colours

- The headline AI Signal now uses four independent layers: **25% forensic A–I evidence, 35% continuous statistical fingerprint, 30% paragraph profile and 10% document regularity**. When several layers are simultaneously elevated, a small corroboration bonus prevents strong multi-layer evidence from being diluted by averaging.
- The statistical fingerprint exposes continuous local metrics for predictability/perplexity proxy, burstiness, token-distribution proxy, repeated n-grams, semantic-density uniformity, repeated syntactic structures, transition concentration and vocabulary diversity. The perplexity/token fields are explicitly labelled proxies in this lightweight deployment because no reference language model is bundled into the Render image.
- **Engine 1 remains detector-independent.** It improves natural scholarly prose without reading the A–I scores.
- **Engine 2 remains the optional API engine** with Terra/Luna selection and the same preservation checks.
- **Engine 3, Signal-Guided rewrite** reads the active A–I detector families and explicitly targets safely editable signals. E (specificity) and H (voice/register) are diagnostic-only when fixing them would require inventing detail or personal voice. Engine 3 reports the targeted A–I score before and after its rewrite.
- Added a **Signal-coloured text** tab. Flagged sentences carry A–I badges and category-specific colours, and clicking a sentence shows the evidence that fired.
- The coloured HTML export now uses the A–I category colours rather than only generic red/orange/yellow risk bands.
- Added a **Statistical fingerprint** tab so users can inspect the continuous metrics behind the statistical layer.
- Source-line-aware segmentation prevents title blocks, author affiliations, form rows and table lines from being misread as giant prose sentences.
