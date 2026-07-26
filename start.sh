#!/usr/bin/env sh
set -e
python -m pip install -r requirements.txt
exec python -m uvicorn app:app \
  --host 0.0.0.0 \
  --port "${PORT:-10000}" \
  --workers "${WEB_CONCURRENCY:-1}" \
  --proxy-headers \
  --forwarded-allow-ips='*'
