---
title: Sensorflow Studio
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: other
license_name: PolyForm-Noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
---

# Sensorflow Studio

An autonomous-driving data platform: dataset exploration, label evaluation,
MITL (model-in-the-loop) review tooling, and the SSAM Safety Dashboard for
statewide surrogate-safety analytics.

## What's served

- `/` — legacy static UI (dataset / training studio)
- `/dashboard/` — React (Vite) frontend, including the SSAM Safety Dashboard
- `/api/...` — FastAPI backend endpoints (e.g. `POST /api/ssam/statewide`)

## Notes for this Space

This is a CPU Space running a slim dependency set. Features requiring GPU or
local services — YOLO training, SAM-assisted annotation, and the Ollama
copilot — are not functional here; the dashboards and analytics APIs are.

Deployed from the project repository via
`deploy/huggingface/deploy.py` (uploads files with `huggingface_hub`; the
frontend is built inside the Docker image).

## License

**PolyForm Noncommercial License 1.0.0** — see [`LICENSE`](LICENSE).

- Source may be viewed and used for **non-commercial** purposes only.
- **Commercial use** requires a separate paid/commercial license from the copyright holder.
- This is a **source-available / non-commercial** license, **not** an OSI Open Source license.

