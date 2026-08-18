# Render deployment checklist

## Recommended deployment path

Use the included `render.yaml` Blueprint. It deploys the app as a Docker-based Render Web Service.

## Before deployment

- Put the project files at the repository root.
- Confirm that `Dockerfile` and `render.yaml` are in the root.
- Do not upload a real `.env` file or API key to Git.
- The Blueprint preconfigures `HUMANIZER_PROVIDER=openai`. Add `OPENAI_API_KEY` as a Render secret if you want Engine 2 to run. Engine 1 and Engine 3 remain available without any API key.

## Render settings

| Setting | Value |
|---|---|
| Runtime | Docker |
| Region | Frankfurt |
| Health check | `/healthz` |
| Public port | Render-provided `PORT` |
| Worker count on Free/Starter | `1` |
| Persistent disk | Not required |

## After deployment

1. Open the `onrender.com` URL.
2. Open `/healthz` and confirm a JSON response with `status: ok`.
3. Paste a short scholarly paragraph and run **Detect AI**.
4. Run **Humanize scholarly text** with **Engine 1, Local rewrite**, then test **Engine 3, Signal-Guided rewrite** on a passage with active A–I signals.
5. Test TXT and DOCX upload and both DOCX exports.
6. Engine 2 is prewired for OpenAI in `render.yaml`: `HUMANIZER_PROVIDER=openai`, `OPENAI_MODEL=gpt-5.6-terra`, and `OPENAI_BASE_URL=https://api.openai.com/v1`. Render will prompt for the secret `OPENAI_API_KEY` because the blueprint marks it `sync: false`. Set `OPENAI_MODEL=gpt-5.6-luna` only if you prefer the economy/high-volume option.

## Production recommendations

- Use a paid instance for regular institutional use or large simultaneous jobs.
- Keep one Uvicorn worker on low-memory instances.
- Scale horizontally for concurrency instead of increasing workers on a small instance.
- Treat an external model API key as a Render secret.
- Inform users that text is sent to the configured external model only when Engine 2, API rewrite is selected.
- Use access control before exposing the service to a restricted institutional audience.


## Version 1.8 detector calibration and evidence-locked rewriting

- The dashboard no longer displays an estimated AI-edited fraction. It shows the **AI Signal Index**, its exact **Human-like Style** complement, **Signal level**, **Forensic category score**, **Active signal categories** and **Evidence items**.
- Signal levels are descriptive rather than authorship claims: Minimal, Low, Moderate, Elevated and Strong AI-style signal.
- A detector-variability notice explains that different AI-writing detectors can disagree substantially on the same polished scholarly passage.
- Engine 1 Deep mode is more active on editable prose, but form rows, names, emails, tables, numbers, citations, references, equations, headings, URLs and DOIs are locked and validated before an edit is accepted.
- Academic calibration excludes common technical uses such as Huber robust regression, ordinary statistical “between X and Y” phrasing and semicolons inside citations from inflated AI-style evidence.
- If Engine 2 is selected without `OPENAI_API_KEY`, the response explicitly reports **Engine 1 fallback**. Add the secret under Render → Environment and redeploy/restart the service for Engine 2 to run.


## Version 2.0 detector independence

- Humanization no longer uses AI Signal as an acceptance condition. A rewrite is accepted only when protected content is preserved and the internal writing-quality metric does not degrade.
- The post-rewrite AI score is recalculated independently and may move up or down. That movement is diagnostic, not a pass/fail gate.
- The detector now reports paragraph-level distribution and a four-component composite score, which makes long manuscripts less vulnerable to whole-document averaging.
- Human-context evidence affects confidence only and is not subtracted from AI-style evidence.


## Version 2.1

- Engine 3 is fully local and requires no additional environment variable.
- The new statistical fingerprint is also local and adds no heavy ML-model dependency to the Render image.
- Engine 2 alone sends rewrite text to the configured external API.
- Static assets use the v2.1.0 cache key.
