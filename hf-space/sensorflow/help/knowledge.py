"""Help knowledge base: page guides + docs snippets + static FAQs.

CPU-friendly: pure Python tokenization / scoring, no embeddings or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from sensorflow.about.catalog import get_about

# ---------------------------------------------------------------------------
# Compact page guides (mirrors hf-space/src/help/pageGuides.ts summaries)
# ---------------------------------------------------------------------------

PAGE_GUIDES: Dict[str, Dict[str, object]] = {
    "command": {
        "title": "Command Center",
        "summary": "Aggregate-first quality cockpit for mega-scale evaluation: cohorts and containers first, annotations only at deepest drill-down.",
        "key_actions": [
            "Switch population / evaluation run / baseline",
            "Generate a population or launch an evaluation run",
            "Drill cohorts, sort containers, compare runs for promotion",
        ],
    },
    "overview": {
        "title": "Overview",
        "summary": "Single-glance pipeline health: live counters, headline quality metrics, stage status and alerts.",
        "key_actions": [
            "Monitor counters and alerts",
            "Bootstrap with a synthetic dataset when empty",
            "Follow evidence links from alerts",
        ],
    },
    "datasets": {
        "title": "Datasets",
        "summary": "Registry of ingested datasets — the pipeline entry point.",
        "key_actions": ["Ingest or select the active dataset", "Inspect per-group quality metrics"],
    },
    "label-generation": {
        "title": "Label Generation",
        "summary": "Auto-labeling queue: raw frames → candidate labels with live throughput.",
        "key_actions": ["Start or re-run label generation", "Monitor queue depth and class mix"],
    },
    "rare-events": {
        "title": "Rare Events",
        "summary": "Anomaly-ensemble mining for the long tail of risky samples.",
        "key_actions": ["Tune detectors and fusion strategy", "Re-run detection and benchmark techniques"],
    },
    "raremine": {
        "title": "Rare-Event Miner",
        "summary": "Multimodal miner for costumed pedestrians — proposals, never verdicts.",
        "key_actions": ["Generate a scene bank", "Run the miner", "Approve or reject candidates into datasets"],
    },
    "quality": {
        "title": "Quality Engine",
        "summary": "GT-free structural validation: geometry, sensor consistency, plausibility.",
        "key_actions": ["Inspect pass/fail distributions per check", "Trace thresholds to the quality policy"],
    },
    "regression": {
        "title": "Regression",
        "summary": "Model-version regression tracking across quality metrics and classes.",
        "key_actions": ["Compare model versions", "Inspect regressed classes and deltas"],
    },
    "rca": {
        "title": "Root Cause Lab",
        "summary": "Staged offline-vs-shadow forensics for regression root causes.",
        "key_actions": ["Walk staged RCA boards", "Compare offline and shadow evidence"],
    },
    "triage": {
        "title": "Triage",
        "summary": "Quality-policy gate routing: AUTO_GRADED / FLAGGED / REJECTED with reasons.",
        "key_actions": ["Review gate lines and failure reasons", "Inspect routing counts"],
    },
    "review": {
        "title": "Human Review",
        "summary": "HITL queue for flagged and sampled labels with multi-sensor evidence.",
        "key_actions": ["Verify, correct, or reject labels", "Use sampling plans for CI-backed metrics"],
    },
    "training": {
        "title": "Training",
        "summary": "Flywheel: verified labels → training jobs → new model versions.",
        "key_actions": ["Select a verified dataset", "Start training", "Watch job metrics"],
    },
    "models": {
        "title": "Models",
        "summary": "Registered model versions and their evaluation / regression status.",
        "key_actions": ["Browse models", "Open regression status for a version"],
    },
    "evaluation": {
        "title": "Evaluation Records",
        "summary": "Per-annotation evaluation evidence browser.",
        "key_actions": ["Search and open evaluation records", "Ask the MITL copilot for advisory analysis"],
    },
    "audit": {
        "title": "Audit",
        "summary": "Process-unit accounting and immutable audit trail of decisions.",
        "key_actions": ["Inspect process units", "Filter audit events"],
    },
    "pipeline": {
        "title": "Pipeline Architecture",
        "summary": "How Sensorflow stages connect from ingest through flywheel.",
        "key_actions": ["Read the architecture diagram", "Jump to related pages"],
    },
    "hillclimb": {
        "title": "Hill Climbing EM",
        "summary": "Adaptive EM development and interview-readiness coaching.",
        "key_actions": ["Run interviews and design lab", "Track readiness scores"],
    },
    "vitis": {
        "title": "Hardware Acceleration",
        "summary": "Vitis Vision HIL: quantization gap, ISP augment, temporal stability (emulated).",
        "key_actions": ["Run HIL / ISP / temporal demos", "Compare CPU vs accelerated paths"],
    },
    "ssam": {
        "title": "SSAM Safety",
        "summary": "Legacy statewide SSAM conflict map and street annotations.",
        "key_actions": ["Query statewide conflicts", "Annotate streets on the map"],
    },
    "safety-odd": {
        "title": "ODD Coverage",
        "summary": "Operational design domain coverage vs required cells.",
        "key_actions": ["Inspect coverage gaps", "Filter ODD dimensions"],
    },
    "safety-gates": {
        "title": "Release Gates",
        "summary": "Deterministic release / stop-ship gates for safety claims.",
        "key_actions": ["Evaluate gate sets", "Read fail reasons"],
    },
    "safety-evidence": {
        "title": "Evidence Package",
        "summary": "Assembled safety evidence package for a release candidate.",
        "key_actions": ["Browse evidence artifacts", "Export package summaries"],
    },
    "safety-ssam": {
        "title": "SSAM Conflicts",
        "summary": "SSAM conflict analysis integrated into the safety layer.",
        "key_actions": ["Inspect TTC/PET conflicts", "Link conflicts to scenarios"],
    },
    "safety-calibration": {
        "title": "Calibration",
        "summary": "Sensor calibration validation across camera–LiDAR extrinsics.",
        "key_actions": ["Run calibration checks", "Inspect residual errors"],
    },
    "safety-discrepancy": {
        "title": "Discrepancy Mining",
        "summary": "Mine disagreements between sensors, graders, or model versions.",
        "key_actions": ["Run discrepancy queries", "Triage high-value disagreements"],
    },
    "safety-scenarios": {
        "title": "Scenario DB",
        "summary": "Structured scenario database for safety evaluation coverage.",
        "key_actions": ["Filter scenarios", "Inspect tags and outcomes"],
    },
    "safety-search": {
        "title": "Semantic Search",
        "summary": "Natural-language / semantic search over safety scenarios.",
        "key_actions": ["Query scenarios in plain language", "Open matching scenario cards"],
    },
    "seqeval": {
        "title": "Sequential Regression",
        "summary": "Anytime-valid sequential regression testing with budgets and pairing.",
        "key_actions": ["Configure sequential tests", "Read six-outcome regression status"],
    },
    "bevfusion": {
        "title": "Perception Engines",
        "summary": "BEV fusion camera+LiDAR perception demos and self-eval.",
        "key_actions": ["Run fusion scenes", "Inspect BEV replay and tracks"],
    },
    "scenario-composer": {
        "title": "Scenario Composer",
        "summary": "Compose synthetic evaluation scenarios for studio workflows.",
        "key_actions": ["Build scenario configs", "Save layouts"],
    },
    "pipeline-builder": {
        "title": "Pipeline Builder",
        "summary": "Visual pipeline composition for studio evaluation flows.",
        "key_actions": ["Connect pipeline nodes", "Persist builder layouts"],
    },
    "my-dashboard": {
        "title": "My Dashboard",
        "summary": "Personalizable studio dashboard of key evaluation widgets.",
        "key_actions": ["Arrange widgets", "Save dashboard layout"],
    },
    "retro": {
        "title": "Retrospective Analyzer",
        "summary": "Evidence-tiered failure retrospectives with safety-case RAG.",
        "key_actions": ["Open a retrospective case", "Review severity and launch policy"],
    },
    "closed-loop-lab": {
        "title": "Closed-Loop Lab",
        "summary": "Closed-loop behavioral evaluation and counterfactual simulation.",
        "key_actions": ["Run closed-loop scenarios", "Inspect validity gates"],
    },
    "launch-readiness": {
        "title": "Launch Readiness",
        "summary": "Agentic misclassification triage and launch-readiness scorecard.",
        "key_actions": ["Run agentic triage", "Read launch verdict and evidence graph"],
    },
    "studio2": {
        "title": "Studio 2.0 Governance",
        "summary": "Control plane: entity registry, release gate matrix, observability funnel.",
        "key_actions": ["Browse registry entities", "Evaluate release gate", "Inspect funnel"],
    },
    "legacy": {
        "title": "Legacy Studio",
        "summary": "Classic YOLO train/infer/SSAM studio embedded for compatibility.",
        "key_actions": ["Use legacy train/infer workflows", "Open full-window legacy UI"],
    },
    "production-readiness": {
        "title": "Production Readiness",
        "summary": "Hardening audit scorecard — honest NOT PRODUCTION READY while Criticals remain.",
        "key_actions": ["Filter findings", "Follow FILE:LINE refs", "Review remediation board"],
    },
    "rotr": {
        "title": "ROTR Control Center",
        "summary": "Right-of-the-road violation detection, attribution, consequence, and flywheel.",
        "key_actions": [
            "Generate a scenario bank and run detection",
            "Inspect attribution and consequence replay",
            "Validate HITL items into the regression suite",
        ],
    },
}

PLATFORM_OVERVIEW = (
    "Sensorflow Studio evaluates machine-generated perception labels at scale. "
    "The core loop: measure every label, gate it against a versioned quality policy, "
    "route the uncertain minority to humans (HITL), and feed verified results back into "
    "training. Aggregates lead; individual annotations are drill-down. Major areas: "
    "Command Center (megaeval), label pipeline (datasets → generation → quality → triage → review), "
    "safety & compliance, perception engines (BEV/seqeval), ROTR, and Studio 2.0 governance."
)

STATIC_FAQS: List[Dict[str, str]] = [
    {
        "id": "faq-what-is-sensorflow",
        "title": "What is Sensorflow Studio?",
        "question": "what is sensorflow studio platform overview purpose",
        "answer": PLATFORM_OVERVIEW,
    },
    {
        "id": "faq-how-to-start",
        "title": "How do I get started?",
        "question": "how to start begin bootstrap first steps empty overview",
        "answer": (
            "On Overview, if counters are zero, click “Generate synthetic dataset & run pipeline”. "
            "Or open Datasets to ingest/select data, then Label Generation → Quality → Triage → Human Review. "
            "For mega-scale aggregates, open Command Center and generate a population, then launch an evaluation run."
        ),
    },
    {
        "id": "faq-nav-tips",
        "title": "Navigation tips",
        "question": "keyboard navigation tips hash url sidebar drawer navigate pages",
        "answer": (
            "Use the left drawer to switch pages. URLs are hash-routed (#/page or #/page/entityId) so "
            "browser back/forward and deep links work. The AppBar (?) opens Help (How it works, Glossary, "
            "Pages, Tips, Docs). The chat bubble answers questions about pages and features. "
            "Page headers include an expandable “About this page” panel."
        ),
    },
    {
        "id": "faq-hitl",
        "title": "When do humans review labels?",
        "question": "human review hitl flagged triage sampling verify correct reject",
        "answer": (
            "Triage routes gate failures to FLAGGED (and hopeless cases to REJECTED). Flagged labels plus "
            "statistically sampled labels enter Human Review. Reviewers verify, correct, or reject with "
            "camera / LiDAR / BEV / temporal evidence. Sampling also puts confidence intervals on headline metrics."
        ),
    },
    {
        "id": "faq-no-gpu",
        "title": "Does the help chatbot need a GPU?",
        "question": "gpu ollama llm offline cpu space chatbot requirements",
        "answer": (
            "No. Help chat uses a local FAQ / page-guide matcher by default and works on CPU Hugging Face Spaces. "
            "If a local Ollama endpoint is reachable it may enrich answers; otherwise you still get deterministic FAQ replies."
        ),
    },
    {
        "id": "faq-quality-vs-eval",
        "title": "Quality Engine vs Evaluation Records",
        "question": "difference quality engine evaluation records gt-free reference",
        "answer": (
            "Quality Engine runs GT-free structural checks (geometry, point support, sensor agreement) on every label. "
            "Evaluation Records show per-annotation measurements when reference GT or graders are available, including "
            "IoU, anomaly scores, and triage decisions."
        ),
    },
    {
        "id": "faq-docs",
        "title": "Where are the docs?",
        "question": "documentation docs architecture prd hardening readme",
        "answer": (
            "In-repo docs live under hf-space/docs/ (architecture, hardening audit, PRDs, retro reports) plus "
            "top-level README / ARCHITECTURE / DASHBOARD_README. The Help menu Docs tab lists the same guides."
        ),
    },
]


def _version_faqs() -> List[Dict[str, str]]:
    """FAQs derived from the About catalog so version numbers stay in sync."""
    about = get_about()
    version = about["version"]
    name = about["name"]
    links = about["links"]
    releases = about["releases"] or []
    latest = releases[0] if releases else {"version": version, "date": "", "title": "", "highlights": []}
    highlights = "; ".join(str(h) for h in (latest.get("highlights") or [])[:6])
    history = "; ".join(f"{r.get('version')} ({r.get('date')}) {r.get('title')}" for r in releases[:8])
    return [
        {
            "id": "faq-version",
            "title": "What version is Sensorflow Studio?",
            "question": "what version current app version about release number v0",
            "answer": (
                f"{name} is version {version}. Click the AppBar v{version} chip or open Help → About. "
                f"GitHub: {links['github']}. Hugging Face Space: {links['hf_space']}."
            ),
        },
        {
            "id": "faq-whats-new",
            "title": "What's new?",
            "question": "what's new whats new recent changes updates latest release changelog",
            "answer": (
                f"Latest release {latest.get('version')} ({latest.get('date')}): {latest.get('title')}. "
                f"{highlights} Open Help → About or the version chip for the full list."
            ),
        },
        {
            "id": "faq-release-notes",
            "title": "Where are the release notes?",
            "question": "release notes changelog versions history each version about page",
            "answer": (
                f"Release notes are listed newest-first on About (Help → About, or the v{version} chip). "
                f"Notable versions: {history}."
            ),
        },
        {
            "id": "faq-about",
            "title": "About Sensorflow Studio",
            "question": "about sensorflow studio product github huggingface space links",
            "answer": (
                f"{about['description']} Current version {version}. "
                f"GitHub: {links['github']}. Hugging Face Space: {links['hf_space']}."
            ),
        },
    
    {
        "id": "faq-load-dataset",
        "title": "How do I load a dataset?",
        "question": "how do i load a dataset ingest select yaml source path browse catalog configuration",
        "answer": (
            "Legacy Studio: open Dataset Configuration, pick a dataset type (or Local), set the source path / "
            "data YAML, then Save. For the 3D pipeline, use Ingest & Fusion with Local and/or vendor stubs "
            "(Alpamayo, Waymo, A2D2), set a sequence id, and run ingest — then browse frames in the pipeline viewer. "
            "React Sensorflow Studio: open Datasets to ingest or select the active dataset (AppBar chip), or on "
            "Overview click “Generate synthetic dataset & run pipeline” when counters are empty."
        ),
    },
    {
        "id": "faq-strict-execution",
        "title": "What is Strict Execution Mode?",
        "question": "what is strict execution mode strict mode ledger evidence verified succeeded refuse incomplete",
        "answer": (
            "Strict Execution Mode refuses to treat a Studio op as SUCCEEDED when evidence is incomplete. "
            "Ops (load, train, infer, grade, auto-label, pipeline stages) write a persistent execution ledger "
            "under runs/executions/. UI PASS badges are not proof — use Evidence cards and the Global Execution "
            "Console. When Strict Mode is on, SUCCEEDED requires verified artifacts; otherwise the run is marked "
            "failed/incomplete even if a process exited 0."
        ),
    },
    {
        "id": "faq-pipeline-stages",
        "title": "How do pipeline stages work?",
        "question": "pipeline stages ingest perception tracking quality gate launch how to use stages workflow",
        "answer": (
            "Typical 3D flow: Ingest & Fusion → Perception (SAM proposals) → Tracking → Quality Gate → Launch Gate. "
            "Each stage writes sequence artifacts under runs/pipeline/. Catalog “100% loaded” is not the same as "
            "browsable frames — use the pipeline frame browser after ingest. Gates can block export when thresholds fail."
        ),
    },
]


STATIC_FAQS.extend(_version_faqs())


@dataclass(frozen=True)
class KnowledgeDoc:
    id: str
    title: str
    kind: str  # page | faq | docs
    text: str
    page_id: Optional[str] = None


def _docs_root() -> Path:
    # Prefer hf-space/docs when present; fall back to repo-root docs/.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "docs",  # hf-space/docs or repo/docs depending on install path
        here.parents[2] / "hf-space" / "docs",  # root sensorflow/help → hf-space/docs
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]



def _snippet_from_markdown(path: Path, max_chars: int = 900) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    lines: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        lines.append(s)
        if sum(len(x) for x in lines) > max_chars:
            break
    return " ".join(lines)[:max_chars]


@lru_cache(maxsize=1)
def build_knowledge_index() -> List[KnowledgeDoc]:
    docs: List[KnowledgeDoc] = []

    docs.append(
        KnowledgeDoc(
            id="platform-overview",
            title="Platform overview",
            kind="faq",
            text=PLATFORM_OVERVIEW,
        )
    )

    for pid, guide in PAGE_GUIDES.items():
        title = str(guide["title"])
        summary = str(guide["summary"])
        actions = "; ".join(str(a) for a in guide.get("key_actions", []))  # type: ignore[arg-type]
        text = f"{title}. {summary} Key actions: {actions}"
        docs.append(
            KnowledgeDoc(
                id=f"page:{pid}",
                title=title,
                kind="page",
                text=text,
                page_id=pid,
            )
        )

    for faq in STATIC_FAQS:
        docs.append(
            KnowledgeDoc(
                id=faq["id"],
                title=faq["title"],
                kind="faq",
                text=f"{faq['question']}. {faq['answer']}",
            )
        )

    root = _docs_root()
    preferred = [
        "PLATFORM_INVENTORY.md",
        "POPULATION_SCALE_ROADMAP.md",
        "hardening/audit.md",
        "architecture/rotr-architecture.md",
        "architecture/studio2-review.md",
        "architecture/nextgen-adr.md",
        "prd/vitis-hil-regression.md",
        "retro/final-report.md",
    ]
    for rel in preferred:
        path = root / rel
        if not path.is_file():
            continue
        snippet = _snippet_from_markdown(path)
        if not snippet:
            continue
        docs.append(
            KnowledgeDoc(
                id=f"docs:{rel}",
                title=f"Docs · {rel}",
                kind="docs",
                text=snippet,
            )
        )

    return docs


def list_page_guides() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for pid, guide in PAGE_GUIDES.items():
        out.append(
            {
                "page_id": pid,
                "title": guide["title"],
                "summary": guide["summary"],
                "key_actions": guide["key_actions"],
            }
        )
    return out
