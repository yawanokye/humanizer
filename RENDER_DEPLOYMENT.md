# Render deployment checklist

## Recommended deployment path

Use the included `render.yaml` Blueprint. It deploys the app as a Docker-based Render Web Service.

## Before deployment

- Put the project files at the repository root.
- Confirm that `Dockerfile` and `render.yaml` are in the root.
- Do not upload a real `.env` file or API key to Git.
- The Blueprint preconfigures `HUMANIZER_PROVIDER=openai`. Add `OPENAI_API_KEY` as a Render secret if you want Engine 2 to run. Engine 1 remains available without any API key.

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
4. Run **Humanize scholarly text** with **Engine 1, Local rewrite** selected.
5. Test TXT and DOCX upload and both DOCX exports.
6. Engine 2 is prewired for OpenAI in `render.yaml`: `HUMANIZER_PROVIDER=openai`, `OPENAI_MODEL=gpt-5.6-terra`, and `OPENAI_BASE_URL=https://api.openai.com/v1`. Render will prompt for the secret `OPENAI_API_KEY` because the blueprint marks it `sync: false`. Set `OPENAI_MODEL=gpt-5.6-luna` only if you prefer the economy/high-volume option.

## Production recommendations

- Use a paid instance for regular institutional use or large simultaneous jobs.
- Keep one Uvicorn worker on low-memory instances.
- Scale horizontally for concurrency instead of increasing workers on a small instance.
- Treat an external model API key as a Render secret.
- Inform users that text is sent to the configured external model only when Engine 2, API rewrite is selected.
- Use access control before exposing the service to a restricted institutional audience.


## Version 1.7 detector/rewrite consistency

- The dashboard counters now report **Active signal categories** and **Evidence items**, which directly explain the document-level AI Signal Index.
- Sentence-level high/moderate counts are no longer used as the headline statistics because document-level signals such as burstiness can exist without a single high-risk sentence.
- Academic calibration excludes common technical uses such as Huber robust regression, ordinary statistical “between X and Y” phrasing and semicolons inside citations from inflated AI-style evidence.
- If Engine 2 is selected without `OPENAI_API_KEY`, the response explicitly reports **Engine 1 fallback**. Add the secret under Render → Environment and redeploy/restart the service for Engine 2 to run.
