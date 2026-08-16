# Render deployment checklist

## Recommended deployment path

Use the included `render.yaml` Blueprint. It deploys the app as a Docker-based Render Web Service.

## Before deployment

- Put the project files at the repository root.
- Confirm that `Dockerfile` and `render.yaml` are in the root.
- Do not upload a real `.env` file or API key to Git.
- Keep `HUMANIZER_PROVIDER=none` for the first deployment so Engine 1 local rewrite remains the only active rewrite path.

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
6. Add Engine 2 model-provider environment variables only after the Engine 1 workflow passes. For OpenAI use `HUMANIZER_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL`. The default is `gpt-5.6-terra`; set `OPENAI_MODEL=gpt-5.6-luna` for the economy/high-volume option. `OPENAI_BASE_URL=https://api.openai.com/v1` is optional.

## Production recommendations

- Use a paid instance for regular institutional use or large simultaneous jobs.
- Keep one Uvicorn worker on low-memory instances.
- Scale horizontally for concurrency instead of increasing workers on a small instance.
- Treat an external model API key as a Render secret.
- Inform users that text is sent to the configured external model only when Engine 2, API rewrite is selected.
- Use access control before exposing the service to a restricted institutional audience.
