# Changelog

Notable Sensorflow Studio versions. Dates are commit dates from git history
(not invented). The in-app About dialog reads the same notes from
`hf-space/src/content/releases.json`.

## [0.5.0] — 2026-08-13

About and versioning visibility.

- About dialog and Help menu tab listing each notable version with release notes
- Current version chip in the AppBar (and a matching chip at the bottom of the nav drawer)
- Shared releases catalog (JSON) plus this changelog; `GET /api/about` and `GET /api/version`
- Help chatbot answers for “what version”, “what’s new”, “release notes”, and “about”

## [0.4.0] — 2026-08-13

In-app help chatbot (`bd54685`, PR #3).

- Help menu: How it works, Glossary, Pages, Tips, and Docs tabs
- Per-page “About this page” guides plus a floating Q&A chatbot
- CPU-friendly FAQ matcher with optional Ollama enrichment

## [0.3.0] — 2026-08-13

Closed-loop, governance, hardening, and ROTR.

- Next-gen closed-loop evaluation with counterfactuals, causal replay, and a launch gauntlet (`/api/nextgen`)
- Studio 2.0 control-plane registry, composed release gate, hardware gate matrix, and observability funnel (`/api/studio2`)
- Production hardening audit: 30 findings, surgical fixes, contracts, HITL prioritization, and a readiness scorecard (`/api/hardening`)
- ROTR (right-of-the-road) violation platform: rule engine, causal attribution, consequence replay, and training flywheel (`/api/rotr`)

## [0.2.0] — 2026-08-12

L4 Studio platform, perception engines, and safety suite.

- Automated 3D perception pipeline and Phase 1 population-scale evaluation foundation (PR #2)
- BEV-Fusion perception engine with masklet temporal propagation and self-evaluation (`/api/bevfusion`)
- L4 label-evaluation + aggregate-first megaeval backends, unified React app shell, Hugging Face Space deploy
- Safety & compliance: ODD coverage, release gates, SSAM, calibration, discrepancy mining, scenario DB (`/api/safety`)
- Sequential regression (seqeval), RCA workbench, rare-event miner, Vitis HIL, Hill Climbing EM, retrospective analyzer, agentic launch readiness, and visual Studio UX

## [0.1.0] — 2026-07-26

Initial auto-labeler and comparative dashboard.

- YOLO auto-labeler framework for detection labeling and quality evaluation
- Crash analysis and data extraction toolkits
- Comparative analytics dashboard visualization layer
- CC BY-NC-SA 4.0 license and DGX Spark deployment prep
