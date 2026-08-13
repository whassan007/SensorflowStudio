/**
 * Page-level explanation content: one entry per screen, rendered by
 * <PageIntro> (header subtitle + expandable "About this page" panel) and
 * reused by the global help menu's page index.
 */
import type { PageId } from '../context/LabelEvalContext';

export interface PageHelpEntry {
  /** 1–2 sentence header subtitle: what this screen shows, where it sits. */
  subtitle: string;
  /** Why this page exists. */
  purpose: string;
  /** How to read the visualizations on it. */
  reading: string;
  /** What the user can do here. */
  actions: string;
  /** How data flows in and out of this stage. */
  dataFlow: string;
}

export const PAGE_HELP: Record<PageId, PageHelpEntry> = {
  command: {
    subtitle:
      'Aggregate-first quality cockpit for mega-scale evaluation runs: population-level metrics, cohorts and containers first — individual annotations only at the deepest drill-down.',
    purpose:
      'At 300k+ objects per run, browsing records is useless. This page inverts the model: it answers "where is the model failing and how badly" from pre-aggregated statistics (the metric cube), and lets you drill Dataset → Cohort → Container → Annotation only where the aggregates point.',
    reading:
      'Hero tiles show headline metrics; when a run has human reviews, a sampling-verified estimate with a 95% CI appears under the automated value. Badges on each panel show query provenance (cache / cube / scan) and latency. "exact" vs "approx" tags mark whether a number is a true count or a sketch estimate. Tabs: Quality (by class, errors, funnel, trend), Cohorts (drill-down explorer with the Why? decomposition), Containers (per-scene table with sort presets), Investigation (error-index search), Compare (model vs model with promotion verdict), Review (statistical sampling with CIs), Shift (train-vs-eval mix), Lineage (reproducibility record).',
    actions:
      'Switch population / evaluation run / baseline in the header. Generate a new population or launch an asynchronous evaluation run (progress streams live). Drill cohorts, sort containers, search errors, run Why? decompositions, build and execute review sampling plans, compare runs for promotion decisions.',
    dataFlow:
      'Populations are generated or ingested upstream; evaluation runs score them against a model version and materialize cube + error index + sketches. Review verdicts flow back into the cube as verified counters, and comparison verdicts feed the Regression page and alerting.',
  },
  overview: {
    subtitle:
      'Single-glance health of the whole label-evaluation pipeline: live counters, headline quality metrics, stage status and active alerts.',
    purpose:
      'The morning-coffee page: is the pipeline running, how much has been processed, are the headline metrics moving, and is anything alerting?',
    reading:
      'Counters track every label\u2019s lifecycle (processed → auto-labeled → auto-graded / flagged → verified / rejected). Quality metrics are computed against the active dataset\u2019s reference GT. Stage cards show each pipeline service\u2019s live state; alert rows carry severity chips — hover any metric or status for its definition.',
    actions:
      'Monitor only — start pipelines from Datasets / Label Generation, and investigate alerts on their source pages (rare events, regression, shift).',
    dataFlow:
      'All numbers stream in live over SSE from the pipeline services. Alerts originate from anomaly detection, regression tracking and distribution-shift checks, and deep-link to their source pages.',
  },
  datasets: {
    subtitle:
      'Registry of ingested datasets and their per-group evaluation results — the entry point of the pipeline.',
    purpose:
      'Select the active dataset, inspect its composition (sensor groups, frame counts, GT availability) and per-group quality metrics.',
    reading:
      'The table lists datasets with GT type and coverage — the confidence chip reflects how trustworthy evaluations against that dataset are (human-verified / gold > vendor > pseudo). Group detail shows precision/recall/F1/IoU per sensor group; hover metric names for exact definitions.',
    actions:
      'Ingest a synthetic dataset, set the active dataset (all other pages follow it), and drill into per-group metrics.',
    dataFlow:
      'Datasets arrive from ingestion (synthetic generator here). Frames flow into Label Generation; evaluation results per group come back from the evaluation engines and are summarized here.',
  },
  'label-generation': {
    subtitle:
      'Auto-labeling stage: the queue that turns raw frames into candidate labels, with live throughput and per-class output mix.',
    purpose:
      'Watch the auto-labeler consume frames from the active dataset, monitor queue depth / throughput / failures, and check the class mix of produced labels.',
    reading:
      'Queue tiles show pending / processing / completed / failed message counts and instantaneous throughput. The class distribution shows what the labeler is emitting — a sudden mix change is an early warning of upstream drift.',
    actions:
      'Start or re-run label generation for the active dataset; monitor progress live.',
    dataFlow:
      'Frames come from the active dataset; produced candidate labels flow to the Quality Engine and evaluation engines for gating and triage.',
  },
  'rare-events': {
    subtitle:
      'Anomaly-ensemble mining for the long tail: find the rare, risky samples worth human attention and training data.',
    purpose:
      'Rare events are where AV perception fails and where training value concentrates. This page configures the detector ensemble, visualizes the population as a needle-in-haystack projection, and benchmarks detector performance.',
    reading:
      'The haystack scatter is a 2D embedding: gray = nominal population, colored = anomaly-flagged candidates; clusters of color indicate systematic blind spots. The four-column config maps detectors → features → fusion strategy → thresholds. Benchmark bars compare per-technique precision/recall/rare-recall on labeled evaluation data.',
    actions:
      'Tune detector parameters (KNN, LOF, Isolation Forest, OC-SVM, DBSCAN), pick the ensemble fusion strategy (max / mean / vote), boost minority classes, re-run detection, and benchmark techniques against each other.',
    dataFlow:
      'Feature vectors come from evaluated labels. Flagged candidates flow into Triage (ANOMALY reason) and the Human Review queue; confirmed rare events are prioritized into Training datasets — the flywheel\u2019s scarcest fuel.',
  },
  raremine: {
    subtitle:
      'Multimodal miner for costumed pedestrians (mascots, inflatables, character suits…) whose distorted silhouettes expose perception weaknesses — proposals, never verdicts.',
    purpose:
      'Costumed pedestrians are the canonical hard rare event: a person whose silhouette no longer looks human. This page runs a deterministic rule-based proposer over a synthetic multimodal scene bank, consolidates multi-frame tracks, removes near-duplicates, and routes candidates through automated + human validation into governed dataset destinations. The separation is strict: the miner proposes, validation measures, humans confirm, statistics judge the miner, and training usage is a governed decision.',
    reading:
      'Each candidate card shows THREE separate confidences (human present / costume present / rare event) — never one collapsed score — plus modality-tagged evidence (the miner cannot cite sensors the scene lacks), alternative hypotheses it kept alive (statue? mannequin? decoration?), and observed model behavior kept strictly apart from the miner\u2019s predicted failure. The track view collapses a 20-frame walk into one event with representative frames. Coverage & curator quality measures the miner itself against the planted truth: precision, recall, calibration, yield and model-value.',
    actions:
      'Generate a scene bank (n, seed), run the miner, filter and review candidates (approve into a dataset destination or reject with a note), inspect lineage and training/eval eligibility, record a governance override to release a protected eval example for training, and re-mine with the improvement report\u2019s suggested config.',
    dataFlow:
      'Scene banks are synthetic and deterministic (planted rare events + confounders give exact ground truth). Approved candidates land in rare-event / hard-example / protected evaluation sets with full lineage; the leakage guard forces training_eligible=false for protected eval sets unless an explicit, audited override exists. Curator metrics feed the next mining run\u2019s configuration.',
  },
  quality: {
    subtitle:
      'GT-free structural validation: geometry, sensor-consistency and plausibility checks that run on every label — no reference truth required.',
    purpose:
      'Most labels have no reference GT. This engine checks each label against physics and sensor evidence instead: box geometry vs class priors, LiDAR point support, ground contact, camera–LiDAR agreement.',
    reading:
      'Each validation panel shows pass/fail distributions per check. Hover any check name for what it measures and its threshold source (quality policy).',
    actions: 'Inspect failing checks and their example labels; thresholds themselves are managed by the quality policy.',
    dataFlow:
      'Labels arrive from Label Generation; per-label check verdicts feed the Triage gates (geometric_validation, sensor_consistency) and are stored as evidence for review.',
  },
  regression: {
    subtitle:
      'Model-version watchdog: tracks per-slice metric deltas across versions and blocks silent quality drops.',
    purpose:
      'Every new model version is compared to its baseline per class and slice, so a regression in pedestrian-at-night cannot hide inside a stable global average.',
    reading:
      'The matrix shows metric deltas per class/slice: green = improved, red = regressed beyond tolerance. Status chips (improved / regressed / baseline) hover-explain the comparison rule that produced them.',
    actions:
      'Review regressed slices, drill into the affected cohorts, and use the verdicts to gate model promotion (see also Command Center → Compare).',
    dataFlow:
      'Evaluation results per model version stream in from the evaluation engines; REGRESSED verdicts raise alerts on Overview and can block auto-grading via the MODEL_REGRESSION policy gate.',
  },
  triage: {
    subtitle:
      'Deterministic decision layer: every evaluated label is auto-graded, flagged for review, or rejected — with full gate-line evidence.',
    purpose:
      'Convert evaluation evidence into an auditable routing decision at machine speed, reserving humans for the cases that actually need them.',
    reading:
      'The decision panel shows status counts and per-reason breakdowns. Every decision lists its gate lines: measured value vs policy threshold vs verdict. Hover a status or failure reason for its definition; the policy ID ties the decision to the exact threshold set used.',
    actions:
      'Inspect decisions and their evidence. Decisions are deterministic — to change outcomes, change the quality policy, not individual rows.',
    dataFlow:
      'Evidence arrives from the evaluation engines (geometry, anomaly, consensus, tracking). AUTO_GRADED labels proceed toward training eligibility; FLAGGED labels enter the Human Review queue with their evidence attached; decisions and policy IDs go to the Audit trail.',
  },
  review: {
    subtitle:
      'Human-in-the-loop review: the prioritized queue of flagged labels, each with camera / LiDAR / BEV / temporal evidence for a verify-correct-reject verdict.',
    purpose:
      'Humans are the scarcest resource in the loop. This queue focuses them on the labels automation could not settle, with all evidence in one place.',
    reading:
      'Queue rows are ordered by severity and safety criticality; each task shows its failure reasons. The detail view renders synchronized camera, LiDAR 3D, bird\u2019s-eye and temporal strips plus the gate lines that flagged it. Hover reasons and metrics for definitions.',
    actions:
      'Claim a task, inspect evidence across views, then verify (label correct), correct (fix geometry/class — becomes new GT), or reject (label wrong). Escalation routes hard cases to senior graders.',
    dataFlow:
      'Tasks arrive from Triage (FLAGGED) and from statistical review sampling. Verdicts return to the pipeline: VERIFIED/corrected labels become training-eligible ground truth and recalibrate automated metrics; rejects feed auto-labeler error analysis.',
  },
  training: {
    subtitle:
      'The flywheel\u2019s closing loop: verified labels become training datasets, new models train on them, and every new model is evaluated back through this platform.',
    purpose:
      'Turn review output into model improvement: assemble verified-only training sets, launch training jobs, and watch loss / rare-recall / safety-recall converge.',
    reading:
      'The flywheel diagram shows the loop stage-by-stage. Job tiles show live loss and recall trajectories; the log viewer streams training output. Only VERIFIED labels are dataset-eligible — hover the eligibility note for why.',
    actions:
      'Build a training dataset from verified labels, configure and launch a training job, monitor progress, and hand the resulting model to Models / Evaluation for scoring.',
    dataFlow:
      'Verified labels flow in from Human Review; trained model versions flow out to the Models registry and are evaluated by Evaluation / Command Center runs — closing the data flywheel.',
  },
  models: {
    subtitle:
      'Model registry: every trained version with its evaluation metrics, promotion state and comparison baseline.',
    purpose: 'One place to see what models exist, how they score, and which one is deployed as baseline.',
    reading:
      'Cards list versions with headline metrics (precision, recall, mAP 3D, safety / rare recall — hover each for definitions). The promotion chip reflects the promotion policy verdict against the baseline.',
    actions: 'Inspect versions, set comparison baselines, and follow deep links into evaluation runs and regression views.',
    dataFlow:
      'Versions register here after Training; evaluation metrics attach from evaluation runs; promotion verdicts come from the compare policy (Command Center → Compare, Regression page).',
  },
  evaluation: {
    subtitle:
      'Record-level evaluation browser: individual evaluation records with their full per-gate evidence — the forensic complement to the aggregate Command Center.',
    purpose:
      'When an aggregate points at a specific label, this page shows that record\u2019s complete evaluation: every engine\u2019s measurements and every gate verdict.',
    reading:
      'Each record lists geometry, sensor, anomaly, consensus and tracking evidence with pass/fail chips. Hover any metric or gate for its definition and threshold source.',
    actions: 'Search / filter records, inspect evidence, follow links into Triage decisions and Review tasks.',
    dataFlow:
      'Records are written by the evaluation engines per label; Triage consumes them for decisions; the Command Center aggregates them into the cube.',
  },
  audit: {
    subtitle:
      'Accountability layer: immutable decision trail plus process-unit consumption — who/what decided, under which policy, at what cost.',
    purpose:
      'Every automated decision must be reconstructible: this page exposes the decision log (label, status, reasons, policy ID, timestamp) and normalized compute cost tracking.',
    reading:
      'Process-unit tiles normalize cost per verified event / per million frames. The trail table is append-only; hover column headers and statuses for definitions.',
    actions: 'Filter/inspect the trail; export for compliance review.',
    dataFlow: 'Entries stream in from Triage and Review verdicts; process units aggregate from every pipeline stage\u2019s reported consumption.',
  },
  pipeline: {
    subtitle:
      'Architecture map of the platform: stages, queues and data flows from raw frames to the training flywheel.',
    purpose:
      'Orientation: see how Input → Evaluation Engines → Quality Gate → Triage → HITL → Training connect, and which services implement each stage.',
    reading:
      'Boxes are stages (live status chips), arrows are data flows. Hover a stage for its role; click through to its page.',
    actions: 'Navigate to any stage\u2019s page; use it as the mental model for the rest of the app.',
    dataFlow: 'This page is the map, not a stage: it visualizes the flows the other pages implement.',
  },
  rca: {
    subtitle:
      'Guided forensic workbench for evaluation discrepancies: when offline says +5% and shadow production says −2%, walk a 13-stage skeptical methodology to find out which number is real — and why.',
    purpose:
      'A metric disagreement has at least eight competing explanations (true regression, distribution shift, feature skew, serving mismatch, label latency, sampling bias, noise, offline contamination). Jumping to the plausible one is how bad models ship and good models die. This lab enforces the full measurement-validity → distribution → parity → significance chain before any conclusion is allowed.',
    reading:
      'The left rail is the methodology: stages unlock strictly in order, and a stage with critical UNKNOWN findings can only be completed by explicitly acknowledging them (recorded forever). The banner keeps the current working-hypothesis set visible — never a single premature conclusion. Each stage screen shows its diagnostic (validity matrix, PSI-badged shift comparisons, the segment heatmap that surfaces Simpson\u2019s paradox, the 2\u00d72 error-transition matrix, the CI-vs-practical-margin plot, ranked feature-parity deltas). Stage 11 is the Root Cause Board: 8 hypotheses scored purely from recorded findings with evidence links back to their stage.',
    actions:
      'Create an investigation (choose a scenario, or a demo with a hidden planted cause for training). Work each stage: review diagnostics, record your own findings, complete or acknowledge-and-proceed. Adjust hypothesis confidences on the board with notes, then export the final report (JSON/markdown) with remediation tiers and the minimum-additional-evidence answer.',
    dataFlow:
      'Investigations generate a paired offline + shadow evaluation dataset (persisted under runs/rca/). Every diagnostic is computed from that data — nothing is invented — and every board score traces to a recorded finding. Reports feed promotion decisions and the recommended follow-up experiments.',
  },
  hillclimb: {
    subtitle:
      'Adaptive Engineering-Manager development & interview-readiness platform: a 4-phase competency blueprint (ML depth → system design → execution & people → hill-climbing simulation) with evidence-based scoring, never self-reported progress.',
    purpose:
      'Interview readiness is a hill-climbing problem: find the competency whose weakness blocks the most downstream skills, fix it with deliberate practice, measure, repeat. This section runs that loop — Assess → Diagnose → Practice → Apply → Evaluate → Improve — over a prerequisite-linked competency graph reconstructed from the Hill Climbing EM blueprint spec.',
    reading:
      'Dashboard bars are per-dimension readiness (Knowledge / Technical Reasoning / Leadership / Execution — tracked separately, never collapsed). The bottleneck callout names the prerequisite blocking the most downstream competencies, not merely the lowest score. Every score in the matrix is explainable: click it to see the evidence artifacts (attempt quotes, STAR stories, design grades, simulation debriefs, interview transcripts) that produced it. Coaching panels quote your own statements as evidence — a score without evidence is never shown.',
    actions:
      'Run the adaptive diagnostic to seed your matrix. Practice generated exercises (structurally different on each retry) and get rubric-based coaching. Diagnose raw experience stories into STAR components with claim-vs-evidence flags. Build architectures in the Design Lab. Play the multi-objective hill-climbing simulation. Take an adaptive mock interview. Review your evidence library and competency matrix.',
    dataFlow:
      'All state persists under runs/hillclimb/. Exercise submissions, STAR stories, design grades, simulation debriefs and interview transcripts become Evidence artifacts that feed the readiness matrix; the matrix feeds bottleneck analysis and the Next Best Action. The Phase-1 "offline +5% / shadow −2%" exercise family cross-links to the Root Cause Lab as the live practice tool.',
  },
  vitis: {
    subtitle:
      'Optional AMD/Xilinx Vitis Vision acceleration layer: an emulated FPGA backend (honestly labeled — no hardware attached) powering HIL quantization-gap regression detection, accelerated ISP + synthetic edge-case generation, and temporal/stereo stability profiling.',
    purpose:
      'Perception pipelines destined for FPGA/ACAP silicon fail in hardware-specific ways: fixed-point quantization drops detections, line-buffer limits create seam artifacts, LUT approximations shift confidences. This page quantifies those gaps BEFORE hardware exists, using a constraint-faithful CPU emulator behind the same VisionBackend interface a real Vitis/XRT backend will implement.',
    reading:
      'The backends panel states honestly which backends exist; the amber EMULATED badge means every FPGA latency/speedup figure is analytically modeled (pixels/cycle × clock), never measured. HIL tab: verdict banner is a sequential (anytime-valid) three-outcome decision; the ablation bars attribute the observed gap to precision vs streaming vs LUT causes; the sweep chart shows gap vs bit-width with the minimal viable config called out. ISP tab: per-stage PSNR chips compare emulated output against the float32 reference; CPU fps is measured wall clock, FPGA fps is modeled. Temporal tab: engines are scored against a model-independent optical-flow motion baseline; the timeline strip highlights (red) frames where a detection dropped while flow proves the object stayed observable; the agreement banner is the meta-check that fixed-point flow does not change verdicts.',
    actions:
      'Configure precision (ap_fixed<W,I>), XFCVDEPTH line-buffer depth, LUT toggles and target device. Run paired HIL comparisons and bit-width sweeps. Run the ISP over both backends and generate evaluation-only augmentation batches with full lineage. Profile engine temporal stability across both backends and inspect per-cohort breakdowns. Read the three PRDs served from docs/prd/.',
    dataFlow:
      'Scenes and engines come read-only from the bevfusion package; sequential verdicts use seqeval\u2019s anytime-valid tests when importable (local paired-t fallback otherwise). Runs persist under runs/vitis/. Generated variants are evaluation-set supplements tagged evaluation-only (never training-eligible) and are offered to the raremine candidate flow when that package is available.',
  },
  ssam: {
    subtitle:
      'Surrogate-safety analysis of real intersections: conflicts ranked by TTC / PET / severity on an interactive map — the safety context the label platform feeds.',
    purpose:
      'Identify dangerous intersections and conflict patterns from trajectory data using surrogate safety measures (no crashes required).',
    reading:
      'Map markers are intersections sized/colored by severity; the drawer ranks conflicts (critical / high / medium / low) with TTC, PET and speeds — hover the column headers for definitions. The grid lists individual conflict events.',
    actions: 'Filter by county / conflict type / severity, inspect intersections, drill into conflict events.',
    dataFlow:
      'Trajectories come from perception output upstream; hotspots identified here define the safety-critical scenarios that evaluation cohorts and rare-event mining prioritize.',
  },
  'safety-odd': {
    subtitle:
      'Interactive coverage map of the Operational Design Domain: which class × condition cells have enough evaluation evidence, and which gaps carry the most risk.',
    purpose:
      'An autonomy system is only validated inside the conditions it was actually evaluated in. This page grids the ODD taxonomy into cells (class × weather/lighting/density/road type), scores each cell\u2019s sample adequacy and performance, and ranks the uncovered cells by exposure-weighted risk so data collection goes where it matters.',
    reading:
      'The heatmap colors each cell by coverage adequacy: green = covered (enough samples, acceptable recall), amber = thin evidence, red = gap, gray = no data. Cell labels show sample counts; click a cell for its full detail (samples vs required, recall, production frequency weight). The gap table below is sorted by risk score = production exposure × severity × uncertainty — the top row is literally "the most dangerous thing we haven\u2019t tested enough".',
    actions:
      'Switch the condition dimension (weather / lighting / density / road type), click cells to inspect, and use "Fill gap" on a gap row to simulate targeted scenario generation for that cell — a before/after dialog shows how the coverage rate and the cell\u2019s adequacy would move.',
    dataFlow:
      'Coverage is computed from the published mega-eval run\u2019s per-cohort sample counts and recall, joined against the ODD taxonomy and production-fleet exposure weights. Fill-gap simulations call the scenario generator preview; actually collecting the data happens upstream in scenario generation / data acquisition.',
  },
  'safety-gates': {
    subtitle:
      'The release decision, made visible: five safety gates evaluated candidate-vs-baseline, each expanding into its actual-vs-threshold checks, with an editable gate policy.',
    purpose:
      'A model ships only if every safety gate passes. This page runs the full gate evaluation (overall quality, safety-critical recall, regression vs baseline, ODD coverage, calibration health) and shows exactly which check blocked a release — no more digging through logs to learn why promotion failed.',
    reading:
      'The verdict banner is the headline: RELEASE_READY (all gates green) or BLOCKED (any gate red). The pipeline strip shows the five gates as connected stages — click one to expand its checks: each row is measured value vs policy threshold with a pass/fail chip. The policy panel lists every threshold with its rationale.',
    actions:
      'Pick candidate and baseline runs, evaluate gates, expand gates to inspect checks, follow the link to the full Evidence Package, and edit gate thresholds in the policy editor (saved server-side; re-evaluate to apply).',
    dataFlow:
      'Gate inputs come from the published evaluation runs (metric cube aggregates, per-cohort recall, calibration diagnostics). The verdict + checks feed the Safety Evidence Package; threshold edits persist to the safety policy store and apply to subsequent evaluations.',
  },
  'safety-evidence': {
    subtitle:
      'The generated Safety Evidence Package for a candidate run: a structured, human-readable compliance document assembled from live evaluation data — clearly labeled as a demonstration artifact.',
    purpose:
      'Safety cases require documented evidence: what was evaluated, under which policy, with which results and known limitations. This page renders the generated evidence package (per ISO 21448 / UL 4600-style structure) so reviewers read one coherent document instead of a dozen dashboards.',
    reading:
      'The document is organized into sections — system description, evaluation scope, gate results, ODD coverage summary, calibration status, known limitations, sign-off placeholders. The prominent disclaimer matters: this is a synthetic demonstration package generated from demo data, not a real certification artifact.',
    actions:
      'Select the candidate run, read the rendered markdown, and download the underlying JSON for archival or external tooling.',
    dataFlow:
      'The package is assembled server-side from gate evaluations, coverage analysis and calibration results for the chosen run. The JSON download is the exact source of the rendered document.',
  },
  'safety-ssam': {
    subtitle:
      'Surrogate-safety conflict analysis for a simulated scenario: every near-miss scored with TTC / PET / DRAC / ΔS / CSI, plotted on a mini conflict map.',
    purpose:
      'Crashes are (thankfully) too rare to measure safety directly — surrogate measures score how close interactions came to conflict. This panel runs the SSAM-style analysis over a scenario\u2019s trajectories and ranks the conflicts it finds, so scenario risk is quantified, not guessed.',
    reading:
      'Aggregate tiles show total conflicts and the worst values (min TTC, min PET, max DRAC, aggregate CSI). Each conflict row carries chips for its measures — hover each chip for the definition (TTC = time-to-collision if paths continue; PET = gap between occupying the same space; DRAC = braking needed to avoid; CSI combines them into severity). The map plots conflict points at their location, colored by severity; the bars compare CSI across conflicts.',
    actions:
      'Choose scenario, random seed and reaction-delay assumption, run the analysis, inspect conflicts on the list and map.',
    dataFlow:
      'Trajectories are simulated deterministically from the scenario definition; conflicts and measures are computed by the safety backend. High-CSI scenario patterns are candidates for the Scenario DB and evaluation cohort design.',
  },
  'safety-calibration': {
    subtitle:
      'Camera–LiDAR extrinsic calibration validation: three scenario modes (clean / miscalibrated / perception failure), a residual scatter with brush-select, and a clear verdict badge.',
    purpose:
      'If the sensors disagree about where things are, every downstream label is suspect — and the failure can masquerade as a model problem. This panel validates cross-sensor geometric consistency and, critically, distinguishes systematic miscalibration (all residuals shifted the same way) from perception failures (a few objects wildly off).',
    reading:
      'The status badge is the verdict: CALIBRATED / MISCALIBRATED / PERCEPTION_FAILURE. Checks table shows each test (mean residual, residual direction consistency, outlier fraction) against thresholds. The scatter plots per-object residuals (camera-vs-LiDAR offset): a tight cloud at the origin = healthy; a shifted cloud = miscalibration; a tight cloud plus far outliers = perception failure. Drag on the chart to brush-select outlier objects and list them.',
    actions:
      'Pick a validation mode (clean baseline, planted rotation miscalibration with adjustable offset, planted perception failures with adjustable tamper fraction), run validation, brush the scatter to inspect object residuals.',
    dataFlow:
      'Object pair residuals are computed from synchronized camera and LiDAR detections in the synthetic scene bank. MISCALIBRATED verdicts should gate labeling runs upstream; the planted modes exist to prove the discriminator works.',
  },
  'safety-discrepancy': {
    subtitle:
      'Cross-source disagreement mining: where auto-labels, human labels and model predictions contradict each other, grouped by type and cohort.',
    purpose:
      'Disagreements between labeling sources are the cheapest quality signal that exists — no new ground truth required. Mining them reveals systematic auto-labeler blind spots, ambiguous classes and cohorts where human review is being wasted or missing.',
    reading:
      'Tiles count total mined discrepancies. The donut splits them by type (missing box, spurious box, class confusion, geometry drift…). The cohort table ranks cohorts by discrepancy rate — a high-rate cohort is where labeling quality is systematically weakest. Sample rows show concrete disagreements with both sides\u2019 values.',
    actions:
      'Run the miner, read the type mix and worst cohorts, drill into samples.',
    dataFlow:
      'The miner joins auto-label, human-review and model-prediction records for the same objects and emits typed discrepancies. High-discrepancy cohorts feed review-sampling plans and auto-labeler improvement work.',
  },
  'safety-scenarios': {
    subtitle:
      'The scenario database: every stored test scenario with severity / type / source filters, free-text search, rare-event import and filtered export.',
    purpose:
      'Scenario-based testing needs a governed library: what scenarios exist, where they came from (mined, synthetic, imported), how severe they are, and which subset a given evaluation should run. This browser is that library\u2019s front door.',
    reading:
      'Each row is a scenario with its type, severity chip, source and description. The count tile tracks the library size. Filters are conjunctive: severity × type × source × text query.',
    actions:
      'Filter and search scenarios, import mined rare events into the DB, export the current filtered set as a JSON bundle for external runners.',
    dataFlow:
      'Scenarios arrive from rare-event mining, SSAM conflict patterns and manual/synthetic authoring. Exported bundles feed simulation runners; scenario selections define evaluation cohorts.',
  },
  'safety-search': {
    subtitle:
      'Concept-level scenario search: type "pedestrian at night in rain" and get ranked scenario/container matches with per-stage scoring explanations and an honest provider badge.',
    purpose:
      'Finding evaluation data by concept — not by exact tag — is how engineers actually think. This search combines a symbolic dimension filter with a reasoning-based scorer and shows its work, so a match is never an unexplained black-box similarity number.',
    reading:
      'The stage breakdown shows the pipeline: symbolic filter (hard dimension constraints parsed from your query) → reasoning scorer (concept relevance) → hybrid blend. Each result card lists matched dimensions as chips, the blended score, and a per-stage explanation of why it ranked where it did. The provider badge states which scoring backend ran (LLM vs deterministic fallback) — results are comparable but the badge keeps provenance honest.',
    actions:
      'Enter a concept query, choose the target corpus (containers or scenarios), toggle LLM scoring, and inspect result explanations.',
    dataFlow:
      'Queries run against scenario/container metadata from the published runs and scenario DB. Selected results deep-link into their browsers; searches are read-only.',
  },
  seqeval: {
    subtitle:
      'Anytime-valid sequential regression testing: the run\u2019s verdict, the per-node decision tables, and the evidence chart showing exactly when — and why — the test stopped.',
    purpose:
      'Fixed-n evaluation wastes budget on obvious cases and under-samples subtle ones. Sequential testing accumulates evidence sample-by-sample and stops the moment a decision is statistically safe, with always-valid error control (no peeking penalty). This dashboard makes that decision process visible instead of burying it in a p-value.',
    reading:
      'The verdict banner states the decision (PROMOTE / BLOCK / INSUFFICIENT_EVIDENCE), which gate fired, and the plain-language reason. THE key visual is the sequential-evidence chart: each line is a node\u2019s log e-value trajectory vs samples used; crossing the upper boundary = evidence of regression (block), the lower = evidence of safety (pass); the vertical marker shows where the run stopped. Decision tables list every tested node with its recall delta + confidence sequence, n used vs effective n, decision chip and a badge on safety-primary nodes (they use stricter thresholds). The regression heatmap locates damage by class × condition. The budget funnel shows planned vs actually-used samples vs full population — the savings are the point.',
    actions:
      'Pick a run, read the verdict and its evidence, hover trajectories for per-step values, and launch new sequential runs (choose a preset planted effect or configure a custom one) from the launcher.',
    dataFlow:
      'Runs are configured and executed by the seqeval backend over paired baseline/candidate evaluations of the synthetic population. Verdicts are designed to gate promotion exactly like the Release Gates regression gate; attribution feeds the regression heatmap.',
  },
  bevfusion: {
    subtitle:
      'Camera-only vs camera+LiDAR BEV-fusion engine comparison — headline deltas, per-cohort explanations, and an interactive top-down canvas replaying one scene\u2019s every detection, fusion and track.',
    purpose:
      'Promoting a new perception engine needs both the statistics (is fused better, where, by how much) and the mechanism (WHY is it better). The comparison card gives the verdict; the BEV canvas shows the physics — depth-uncertain camera ellipses, tight LiDAR crosses, and the fused tracker carrying identity through occlusions.',
    reading:
      'The comparison card: green/red chips are candidate-minus-baseline deltas per metric (direction-aware — position error going down is green), the badge is the promotion policy verdict, and blockers list any gate the candidate failed. Cohort bars pair baseline (gray) vs candidate (colored), sorted by delta; hover for the causal explanation. On the canvas: white outlines = ground truth (dashed amber = camera-occluded), orange dots with ellipses = camera detections (the ellipse stretches along the viewing ray — monocular depth uncertainty), teal crosses = LiDAR, filled boxes = fused tracks (dashed purple = masklet-propagated through a detection gap), red outlines = what camera-only would have produced.',
    actions:
      'Run a comparison (sequences / frames / seed), toggle canvas layers, scrub or play frames, hover any object for details, click a fused box to follow its track across frames — watch the dashed purple propagation carry it through occlusion.',
    dataFlow:
      'Scenes are generated deterministically (seeded), both engines label every frame, and the evaluator scores them against ground truth. The canvas replays the same seeds through the same code path server-side, so what you see is exactly what was scored. Comparison reports persist under runs/bevfusion/.',
  },
  'scenario-composer': {
    subtitle:
      'WYSIWYG scenario editor: drag actors from a palette onto a top-down canvas, shape their trajectories, set the environment, and send the composed scene through evaluation.',
    purpose:
      'Authoring test scenarios by editing JSON is slow and error-prone. Here you place pedestrians, vehicles, cones and construction zones directly on a bird\u2019s-eye canvas, drag them (and their trajectory waypoints) until the scene is right, and the machine-readable recipe writes itself.',
    reading:
      'The canvas is ego-centric: the blue triangle is the ego vehicle, range rings mark distance. Each actor shows its footprint and, when selected, its trajectory waypoints. The environment toggles (day/night, clear/rain/fog) re-render the canvas look so lighting/weather context is visible, not abstract. The recipe panel on the right is the live JSON — it updates with every drag.',
    actions:
      'Drag actors from the palette onto the canvas; drag to reposition; select an actor to edit its properties (class, speed, heading) in the inspector and drag its waypoints; toggle environment; then "Run through evaluation" to send the recipe to the scene-consuming backend. Save/load compositions locally.',
    dataFlow:
      'Compositions persist in your browser (and via the studio-ux backend when available). The run action posts the recipe to the best available scene-consuming API — the capabilities probe hides the button when none is reachable.',
  },
  'pipeline-builder': {
    subtitle:
      'Direct-manipulation node-graph editor over the platform\u2019s pipeline architecture: drag stages, re-wire edges, inspect configs — a clearly-labeled what-if draft, not a deployment tool.',
    purpose:
      'The Pipeline Architecture page shows the pipeline as-built; this builder lets you manipulate a DRAFT copy of that graph — move stages, add or remove connections, toggle stages on/off — and read a live "what will run" summary of your draft. It is an explorer and design surface: the real pipeline is read-only and is never modified.',
    reading:
      'Nodes are pipeline stages (colored by role; live status from the real pipeline where available). Edges are data flows — drag from a node\u2019s output port to another node\u2019s input to connect. The summary panel recomputes the execution order (topological), flags orphaned stages and cycles, and diffs your draft against the as-built pipeline. The DRAFT banner is honest: nothing you do here executes.',
    actions:
      'Drag nodes to rearrange; draw edges port-to-port; click a node for its config in the inspector (enable/disable, notes); delete edges; reset to the as-built graph; export the draft as JSON.',
    dataFlow:
      'The initial graph loads read-only from the pipeline architecture API. Drafts live in your browser (and the studio-ux store when available); export produces a JSON description for humans or future tooling.',
  },
  'my-dashboard': {
    subtitle:
      'Your own composed dashboard: drag live widgets from every corner of the platform onto a grid, arrange and resize them, and the layout persists.',
    purpose:
      'Every role watches a different mix of signals. Instead of one fixed overview, this page lets you compose your own: pick widgets (seqeval verdict, ODD coverage mini-map, engine comparison, review queue, gate status…), place and size them on a grid, and keep that layout across sessions.',
    reading:
      'In view mode widgets are live — each one fetches its real data source and renders the same numbers as its home page, with a link to jump there. In edit mode, widgets get drag handles and resize grips; the palette lists available widgets grouped by source page.',
    actions:
      'Toggle edit mode, add widgets from the palette, drag to reposition, resize with the corner grip, remove with the ×, and reset to the starter layout. Layouts save automatically (locally, plus server-side when the studio-ux backend is up).',
    dataFlow:
      'Widget data comes live from the same APIs as the source pages (read-only). Layout state persists to localStorage and mirrors to /api/studio-ux/layout when available, so it can follow you across browsers.',
  },
  retro: {
    subtitle:
      'Agentic failure retrospectives: perception failures become a traceable evidence chain — observed facts, derived metrics, retrieved requirements, AI hypotheses — ending in a deterministic policy-gated launch recommendation.',
    purpose:
      'A false negative on a pedestrian and a phantom brake on a plastic bag need very different engineering responses. This page runs an evidence-tiered retrospective: deterministic code computes the safety metrics and owns the launch decision; the LLM only interprets, correlates and hypothesizes — and every claim carries its evidence tier.',
    reading:
      'The evidence chain flows top-to-bottom from raw failure to human decision; items are color-coded by tier (FACT / DERIVED / RETRIEVED / AI HYPOTHESIS / DETERMINATION). Retrieved standards are SYNTHETIC demonstration documents (badged) — never real standard text. UNKNOWN means the telemetry was absent, never guessed. The hardware card reports honestly whether vLLM can run on this machine.',
    actions:
      'Pick a failure fixture and inference backend (mock is deterministic and always available; Ollama when a local server is running; vLLM only on CUDA/ROCm hosts), run the analysis, inspect the chain / scorecard / citations / tool-call audit trail, and reload prior analyses.',
    dataFlow:
      'Fixtures live in sensorflow/retro/fixtures; analyses and audit trails persist under runs/retro/. Metric math reuses the SSAM safety extensions; distribution stats delegate to the mega-eval engine; the safety-case corpus is indexed at startup (chromadb or deterministic fallback).',
  },
  'launch-readiness': {
    subtitle:
      'Agentic launch readiness & misclassification triage: a five-layer pipeline (detection → evidence → analysis → decision → flywheel) where AI agents only recommend and a deterministic, versioned policy engine plus recorded human review own every launch decision.',
    purpose:
      'A pedestrian misclassified as a construction cone at "~0.01%" is not actionable until the denominator, confidence interval, significance, safety criticality, concentration and novelty are established. This page runs that triage end-to-end and shows exactly which claims are OBSERVED measurements versus AI hypotheses versus missing evidence.',
    reading:
      'The failure queue lists detected patterns with severity (S0–S5) and policy-outcome chips. Inside an investigation: the Evidence Graph colors each node by status (OBSERVED / DERIVED / HYPOTHESIS / UNAVAILABLE); the statistical panel shows Wilson CIs, exact-binomial significance and the seqeval anytime-valid decision with small-sample flags; the safety chain marks links without replay evidence as UNCERTAIN; the concentration heatmap colors strata by relative risk. Every panel is badged AI ANALYSIS (ADVISORY) or DETERMINISTIC. The decision tab shows the four-way policy outcome, the matrix row that fired, and option cards where safety-infeasible options are excluded no matter how cheap.',
    actions:
      'Scan for failures, run the seeded pedestrian→cone worked example, step the five pipeline stages, record human-review decisions (the only path to validation), export the leadership scorecard as JSON, and evaluate candidates against the flywheel-generated suites.',
    dataFlow:
      'Detection scans a seeded synthetic campaign (bevfusion scenes + a paired rate population). Statistics delegate to seqeval; safety replays reuse the SSAM extensions; Option C checks use the ODD taxonomy. Validated failures feed evaluation suites (contamination-guarded) under runs/agentic/, and every action lands in a hash-chained append-only audit log.',
  },
  legacy: {
    subtitle:
      'The original vanilla-JS studio, embedded unchanged — kept for workflows not yet migrated to the React platform.',
    purpose: 'Access legacy tooling during the migration period.',
    reading: 'The iframe below is the old UI as-is; features may overlap with the new pages.',
    actions: 'Use legacy workflows; open in a new tab for full-window work.',
    dataFlow: 'Talks to the same backend and datasets as the new platform.',
  },
};

/** Short one-liner per page for the global help menu index. */
export function pageOneLiner(id: PageId): string {
  return PAGE_HELP[id].subtitle;
}
