/**
 * Central glossary: every metric, status, and concept shown anywhere in the UI
 * is defined once here and rendered identically everywhere via <Term> /
 * <InfoTip> (src/components/help/InfoTip.tsx) and the global help menu.
 *
 * Entry fields:
 *   term    display name
 *   short   one-line definition (used in dense tooltips)
 *   detail  how it is computed / what it means operationally
 *   caveat  honesty note: limits, approximations, failure modes (optional)
 */

export interface GlossaryEntry {
  term: string;
  short: string;
  detail: string;
  caveat?: string;
  /** Category used by the glossary browser in the help menu. */
  category: GlossaryCategory;
}

export type GlossaryCategory =
  | 'Detection metrics'
  | 'Geometry & LiDAR'
  | 'Tracking'
  | 'Anomaly & rarity'
  | 'Safety (SSAM)'
  | 'Human grading & sampling'
  | 'Aggregation & scale'
  | 'Pipeline & statuses'
  | 'Failure reasons'
  | 'Ground truth'
  | 'Operations'
  | 'EM readiness'
  | 'Hardware acceleration';

export const GLOSSARY: Record<string, GlossaryEntry> = {
  // ------------------------------------------------------------ detection metrics
  precision: {
    term: 'Precision',
    category: 'Detection metrics',
    short: 'Of all labels the system produced, the fraction that are correct.',
    detail: 'TP / (TP + FP). A true positive is a predicted box matched to a reference object at 3D IoU ≥ 0.5. High precision means few phantom / wrong labels.',
    caveat: 'Only as trustworthy as the reference: against pseudo-GT, "correct" means "agrees with the auto-label consensus", not human truth.',
  },
  recall: {
    term: 'Recall',
    category: 'Detection metrics',
    short: 'Of all real objects present, the fraction the system labeled.',
    detail: 'TP / (TP + FN). A false negative is a reference object with no matching predicted box at 3D IoU ≥ 0.5. Low recall means missed objects — the most dangerous failure mode for AV perception.',
    caveat: 'Recall against pseudo-GT under-counts misses that the auto-labeler also failed to see.',
  },
  safety_recall: {
    term: 'Safety recall',
    category: 'Detection metrics',
    short: 'Recall computed only over safety-critical objects.',
    detail: 'Same TP/(TP+FN) formula, restricted to objects flagged safety-critical (vulnerable road users, near-path objects, low-TTC actors). Held to a stricter promotion threshold than headline recall.',
    caveat: 'Small denominators: per-cohort safety recall can swing on a handful of objects — check n before reacting.',
  },
  f1: {
    term: 'F1',
    category: 'Detection metrics',
    short: 'Harmonic mean of precision and recall.',
    detail: '2·P·R / (P + R). A single balance number; useful for ranking but hides which side (misses vs phantoms) is failing — always drill into P and R separately.',
  },
  map_3d: {
    term: 'mAP (3D)',
    category: 'Detection metrics',
    short: 'Mean Average Precision over classes using 3D IoU matching.',
    detail: 'For each class, average precision is the area under the precision–recall curve sweeping the confidence threshold; mAP is the mean across classes. Matching uses 3D IoU ≥ 0.5.',
    caveat: 'Sensitive to class balance: rare classes (motorcycle, bus) move mAP as much as vehicle despite far fewer instances.',
  },
  rare_recall: {
    term: 'Rare recall',
    category: 'Detection metrics',
    short: 'Recall restricted to rare-event objects.',
    detail: 'Recall over objects in scenarios tagged rare (near-miss, extreme TTC, unusual classes / poses). The main target metric of the rare-event mining flywheel.',
  },
  confidence: {
    term: 'Confidence',
    category: 'Detection metrics',
    short: 'The model\u2019s own probability that a predicted label is correct.',
    detail: 'Raw score in [0, 1] emitted by the detection head. Used by the quality gate (min_confidence) and by review sampling to build the low/mid/high-confidence strata.',
    caveat: 'Calibrated only approximately: a 0.9 does not mean 90% empirical correctness in every cohort.',
  },
  anomaly_rate: {
    term: 'Anomaly rate',
    category: 'Detection metrics',
    short: 'Fraction of labels flagged anomalous by the ensemble detectors.',
    detail: 'Count of objects whose anomaly score exceeds the policy threshold, divided by all evaluated objects.',
  },

  // ------------------------------------------------------------ geometry & LiDAR
  iou_3d: {
    term: '3D IoU',
    category: 'Geometry & LiDAR',
    short: 'Volume overlap between predicted and reference 3D boxes.',
    detail: 'Intersection volume / union volume of the two oriented boxes, in [0, 1]. IoU ≥ 0.5 counts as a match; mean IoU over matched pairs measures how tightly boxes fit.',
    caveat: 'A box can match at IoU 0.5 and still have significant heading or size error — see orientation / dimension error.',
  },
  position_error: {
    term: 'Position error',
    category: 'Geometry & LiDAR',
    short: 'Distance in meters between predicted and reference box centers.',
    detail: 'Euclidean distance between 3D box centroids. Gated per quality policy (default max 0.5 m); large values indicate localization drift.',
  },
  orientation_error: {
    term: 'Orientation error',
    category: 'Geometry & LiDAR',
    short: 'Heading angle difference in degrees between predicted and reference box.',
    detail: 'Absolute yaw difference, wrapped to [0°, 180°]. Errors near 180° usually mean a flipped heading, which corrupts downstream velocity estimates.',
  },
  dimension_error: {
    term: 'Dimension error',
    category: 'Geometry & LiDAR',
    short: 'How far box length/width/height deviate from the class-typical size.',
    detail: 'Relative deviation of box dimensions vs the class prior (e.g. a “pedestrian” box 3 m wide fails). Used as a geometric validation check.',
  },
  point_density: {
    term: 'Point density',
    category: 'Geometry & LiDAR',
    short: 'LiDAR points per cubic meter inside the labeled box.',
    detail: 'Points-in-box divided by box volume. Very low density means the label rests on little sensor evidence — common at long range or under occlusion.',
  },
  points_in_box: {
    term: 'Points in box',
    category: 'Geometry & LiDAR',
    short: 'Absolute number of LiDAR returns inside the 3D box.',
    detail: 'Boxes with fewer points than the policy minimum are flagged INSUFFICIENT_POINT_SUPPORT: there is not enough evidence to verify the object geometrically.',
  },
  point_in_box_ratio: {
    term: 'Point-in-box ratio (occupancy)',
    category: 'Geometry & LiDAR',
    short: 'Fraction of the box volume actually supported by LiDAR returns.',
    detail: 'Occupancy of the box by points relative to expectation for the class. Low occupancy with a large box suggests the box is oversized or misplaced.',
  },
  ground_contact: {
    term: 'Ground contact',
    category: 'Geometry & LiDAR',
    short: 'Whether the box bottom sits plausibly on the road surface.',
    detail: 'Vertical distance from the box bottom face to the estimated ground plane. Floating or buried boxes fail geometric validation.',
  },
  sensor_consistency: {
    term: 'Sensor consistency',
    category: 'Geometry & LiDAR',
    short: 'Agreement between camera and LiDAR evidence for the same object.',
    detail: 'The 3D box is projected into the camera image and compared to 2D detections. Disagreement (object in LiDAR but not camera, or vice versa) raises SENSOR_DISAGREEMENT.',
    caveat: 'Expected to degrade in rain/fog/night-glare; disagreement there is a weaker error signal.',
  },

  // ------------------------------------------------------------ tracking
  idf1: {
    term: 'IDF1',
    category: 'Tracking',
    short: 'F1 over identity assignments: how consistently the same object keeps the same track ID.',
    detail: 'Matches predicted and reference trajectories, then computes F1 over correctly-identified detections. Penalizes both ID swaps and missed coverage.',
  },
  id_switch: {
    term: 'ID switch',
    category: 'Tracking',
    short: 'A track changed identity mid-trajectory.',
    detail: 'The same physical object was assigned a new track ID (or two objects swapped IDs). Corrupts velocity history and intent prediction; gated as a hard failure.',
  },
  fragmentation: {
    term: 'Track fragmentation',
    category: 'Tracking',
    short: 'One object\u2019s trajectory broken into multiple short tracks.',
    detail: 'Count (or rate) of interruptions where tracking lost then re-acquired the object. High fragmentation usually accompanies occlusion or low point density.',
  },
  track_quality: {
    term: 'Track quality',
    category: 'Tracking',
    short: 'Composite 0–1 score of trajectory smoothness and consistency.',
    detail: 'Combines fragmentation, ID stability and kinematic plausibility (no teleports, physical accelerations). Below policy threshold ⇒ TRACK_FRAGMENTATION flag.',
  },

  // ------------------------------------------------------------ anomaly & rarity
  anomaly_score: {
    term: 'Anomaly score',
    category: 'Anomaly & rarity',
    short: 'Ensemble estimate (0–1) of how unusual this label is vs the population.',
    detail: 'Score fused from KNN distance, Local Outlier Factor, Isolation Forest, One-Class SVM and DBSCAN outlier flags over geometric + contextual features. Above the policy threshold ⇒ flagged ANOMALY.',
    caveat: 'Anomalous ≠ wrong: genuinely rare-but-correct labels also score high — that is exactly what rare-event mining exploits.',
  },
  ensemble_strategy: {
    term: 'Ensemble strategy',
    category: 'Anomaly & rarity',
    short: 'How the individual detector scores are fused into one anomaly score.',
    detail: 'max = most sensitive (any detector can flag); mean = balanced; vote = majority of detectors must agree (most conservative). Choose based on tolerance for false alarms vs missed anomalies.',
  },
  rarity_score: {
    term: 'Rarity score',
    category: 'Anomaly & rarity',
    short: 'How under-represented this sample\u2019s scenario is in the training corpus.',
    detail: 'Inverse frequency of the sample\u2019s cohort (class × weather × lighting × scenario) in existing training data, blended with the anomaly score. High rarity ⇒ high value for training.',
  },
  haystack: {
    term: 'Needle-in-haystack view',
    category: 'Anomaly & rarity',
    short: 'Projection of the whole population where rare events light up against the nominal mass.',
    detail: '2D embedding (feature projection) of labels: gray points are nominal, colored points are anomaly-flagged / rare candidates. Clusters of colored points suggest a systematic blind spot, not a one-off.',
  },

  // ------------------------------------------------------------ safety (SSAM)
  ttc: {
    term: 'TTC (Time to Collision)',
    category: 'Safety (SSAM)',
    short: 'Seconds until two road users would collide on current trajectories.',
    detail: 'Computed from relative position and velocity at the conflict point. Lower is more dangerous; conflicts under ~1.5 s are typically classified critical.',
  },
  pet: {
    term: 'PET (Post-Encroachment Time)',
    category: 'Safety (SSAM)',
    short: 'Gap in seconds between one road user leaving a conflict area and another entering it.',
    detail: 'Unlike TTC it does not require projected collision courses — it measures how narrowly paths actually missed each other. Small PET = near miss.',
  },
  severity_index: {
    term: 'Severity index',
    category: 'Safety (SSAM)',
    short: 'Composite ranking of how dangerous a surrogate-safety conflict is.',
    detail: 'Combines TTC/PET, speeds, masses/classes of participants and conflict geometry into one score used to rank intersections and events (critical / high / medium / low).',
  },

  // ------------------------------------------------------------ human grading & sampling
  grader_consensus: {
    term: 'Grader consensus',
    category: 'Human grading & sampling',
    short: 'Agreement level among independent graders of the same label.',
    detail: 'Fraction of graders agreeing with the majority verdict. Below the policy threshold ⇒ GRADER_DISAGREEMENT and the label routes to senior review.',
  },
  cohens_kappa: {
    term: "Cohen's Kappa",
    category: 'Human grading & sampling',
    short: 'Agreement between two graders, corrected for chance.',
    detail: '(observed agreement − chance agreement) / (1 − chance agreement). 0 = no better than chance, 1 = perfect. Rule of thumb: > 0.6 substantial, > 0.8 near-perfect.',
    caveat: 'Only defined for exactly two raters; distorted when one verdict dominates (prevalence problem).',
  },
  fleiss_kappa: {
    term: "Fleiss' Kappa",
    category: 'Human grading & sampling',
    short: 'Chance-corrected agreement generalized to many graders.',
    detail: 'Extends Cohen\u2019s Kappa to a fixed panel of ≥ 3 raters rating each item. Used for the grader-pool health metrics.',
  },
  krippendorff_alpha: {
    term: "Krippendorff's Alpha",
    category: 'Human grading & sampling',
    short: 'The most general agreement coefficient: any #raters, missing data, any scale.',
    detail: 'Based on observed vs expected disagreement; handles graders who skip items. Preferred when the grading matrix is sparse. α ≥ 0.8 is the conventional reliability bar.',
  },
  wilson_ci: {
    term: 'Wilson 95% CI',
    category: 'Human grading & sampling',
    short: 'Confidence interval for a proportion that stays honest at small n and extreme p.',
    detail: 'Wilson score interval: solves the normal approximation for the true proportion instead of centering on the sample value. Unlike the naive ±1.96·SE interval it never leaves [0, 1] and behaves at p ≈ 0 or 1.',
    caveat: 'A 95% CI means: under repeated sampling, ~95% of such intervals cover the truth — not "95% probability" for this one interval.',
  },
  stratified_sampling: {
    term: 'Stratified risk-weighted sampling',
    category: 'Human grading & sampling',
    short: 'Review budget split across risk strata so rare, risky slices are measured, not just the easy mass.',
    detail: 'The population is partitioned into strata (e.g. low/mid/high confidence × safety-critical); each stratum gets reviews proportional to size × risk weight (Neyman-style). Per-stratum Wilson CIs combine into one population estimate.',
    caveat: 'Estimates are unbiased only if reviews within each stratum are randomly drawn — never cherry-pick inside a stratum.',
  },
  review_sample: {
    term: 'Review sample',
    category: 'Human grading & sampling',
    short: 'The statistically selected subset of labels actually sent to human reviewers.',
    detail: 'Drawn from the candidate pool by stratified risk-weighted sampling. Its verdicts calibrate the automated headline metrics with confidence intervals — reviewing everything at this scale is impossible.',
  },
  candidate_pool: {
    term: 'Candidate pool',
    category: 'Human grading & sampling',
    short: 'All labels eligible for human review, ranked by risk before sampling.',
    detail: 'Union of error-index hits, low-confidence labels, anomalies and safety-critical objects. The sample is drawn from this pool, not from the raw population.',
  },
  sampling_verified: {
    term: 'Sampling-verified',
    category: 'Human grading & sampling',
    short: 'This number was calibrated by human review of a statistical sample.',
    detail: 'The headline (automated) value is shown alongside a human-verified estimate with a 95% CI, derived from stratified review. Where they disagree, trust the interval.',
  },

  // ------------------------------------------------------------ aggregation & scale
  metric_cube: {
    term: 'Metric cube',
    category: 'Aggregation & scale',
    short: 'Pre-aggregated sufficient statistics (TP/FP/FN/IoU sums…) per dimension combination.',
    detail: 'During an evaluation run each partition emits partial counts per (class, weather, lighting, road type, scenario, sensor, distance, speed, occlusion) cell; the reduce step materializes the cube. Any filter/group-by query then reads cell sums instead of scanning records — that is why dashboard queries take milliseconds over 300k+ objects.',
    caveat: 'Cube answers are exact for additive stats (counts, sums). Non-additive stats (quantiles, distinct counts) come from sketches and are approximate.',
  },
  query_source: {
    term: 'Query source (cache / cube / scan)',
    category: 'Aggregation & scale',
    short: 'Where this panel\u2019s numbers came from, and how fast.',
    detail: 'cache = identical query answered from the in-memory result cache (~0.1 ms). cube = aggregated from the metric cube (~10 ms). scan = fell back to reading raw partitions (slow; only for forensic drill-down).',
  },
  hll: {
    term: 'HyperLogLog (HLL)',
    category: 'Aggregation & scale',
    short: 'Approximate distinct-count sketch: huge cardinalities in a few KB.',
    detail: 'Hashes each item and tracks maximum leading-zero patterns per register; the harmonic mean of registers estimates distinct count. Standard error ≈ 1.04/√m (≈ 2% here). Sketches merge losslessly across partitions.',
    caveat: 'Approximate by design — shown next to the exact count when both exist, and always labeled "approx".',
  },
  quantile_sketch: {
    term: 'Quantile histogram (sketch)',
    category: 'Aggregation & scale',
    short: 'Compact histogram that answers percentile queries without keeping raw values.',
    detail: 'Fixed-bin histogram maintained per partition and merged at reduce time; p10/p50/p90 are interpolated from bin counts. Error is bounded by bin width.',
    caveat: 'Percentiles are approximate (bin-resolution); exact tails may differ slightly.',
  },
  reservoir_sampling: {
    term: 'Reservoir sampling',
    category: 'Aggregation & scale',
    short: 'Keeps a fixed-size uniform random sample from a stream of unknown length.',
    detail: 'Each new item replaces a random slot with probability k/n. Used to retain representative exemplar objects per cohort for drill-down without storing everything.',
  },
  approx_vs_exact: {
    term: 'Approx vs exact',
    category: 'Aggregation & scale',
    short: 'Every number is labeled by provenance: exact (cube/counts) or approx (sketch/sample-derived).',
    detail: 'Counts and additive aggregates are exact. Distinct counts (HLL), percentiles (quantile sketches) and human-review estimates (sampling + CI) are approximate, and the UI marks each with an "approx" tag plus this explanation.',
  },
  cohort: {
    term: 'Cohort',
    category: 'Aggregation & scale',
    short: 'A population slice defined by dimension values, e.g. pedestrian × night × rain.',
    detail: 'Cohorts are the primary unit of analysis: metrics are compared across cohorts to find where the model fails, before ever looking at individual annotations.',
  },
  container: {
    term: 'Container',
    category: 'Aggregation & scale',
    short: 'A physical grouping of annotations — one scene/segment captured together.',
    detail: 'Sits between cohort and annotation in the drill-down: Dataset → Cohort → Container → Annotation. Container-level aggregates (per-scene precision/recall/risk) localize failures to specific captures.',
  },
  evaluation_run: {
    term: 'Evaluation run',
    category: 'Aggregation & scale',
    short: 'A first-class asynchronous job that evaluates one model version over one population.',
    detail: 'Lifecycle: created → queued → running (workers emit partial stats per partition) → reducing → materializing (cube + error index + sketches) → published. Only published runs are queryable.',
  },
  population: {
    term: 'Population',
    category: 'Aggregation & scale',
    short: 'The immutable, versioned set of objects an evaluation run scores.',
    detail: 'Generated / ingested once, stored as partitioned columnar files, then referenced by ID. Immutability is what makes runs reproducible and comparable.',
  },
  lineage: {
    term: 'Lineage',
    category: 'Aggregation & scale',
    short: 'The complete recipe to reproduce a run bit-for-bit.',
    detail: 'Dataset version, model version + checkpoint hash, label version, evaluator code version, metric version, threshold config, sampling config, seed, hardware, timestamp. Identical inputs + seed ⇒ identical outputs.',
  },
  distribution_shift: {
    term: 'Distribution shift',
    category: 'Aggregation & scale',
    short: 'The evaluation population\u2019s cohort mix differs from what the model was trained on.',
    detail: 'Train vs eval share is compared per cohort; cohorts whose share grew materially AND whose recall lags the overall value are flagged — the model is being asked about conditions it rarely saw.',
    caveat: 'Shift is a risk signal, not proof of failure: confirm with the cohort\u2019s actual metrics.',
  },
  risk_score: {
    term: 'Risk score',
    category: 'Aggregation & scale',
    short: 'Container-level priority: how likely this scene hides real, important errors.',
    detail: 'Blends error density (FN/FP counts), safety-critical presence, anomaly rate and low review coverage into a 0–1 rank used by the "Highest risk" sort and review candidate pool.',
  },
  error_index: {
    term: 'Error index',
    category: 'Aggregation & scale',
    short: 'A dedicated index of every potential error, searchable by multiple criteria.',
    detail: 'Each FN / FP / localization / anomaly / low-confidence hit is indexed with its dimensions, severity and container, so "worst-N containers where class=pedestrian AND lighting=night" is a lookup, not a scan.',
  },
  embedding_similarity: {
    term: 'Similarity search',
    category: 'Aggregation & scale',
    short: 'Finds containers structurally similar to a query container.',
    detail: 'Cosine similarity over 32-dimensional structural embeddings (dimension mix, error profile), hybridized with hard filters on scenario/lighting. Used to grow one found failure into its whole family.',
    caveat: 'Embeddings are compact projections — treat results as candidates to inspect, not proof of equivalence.',
  },

  // ------------------------------------------------------------ pipeline & statuses
  quality_policy: {
    term: 'Quality policy',
    category: 'Pipeline & statuses',
    short: 'The versioned set of thresholds every label is gated against.',
    detail: 'Defines min IoU, max position/orientation error, min points-in-box, min confidence, min consensus, anomaly threshold, tracking requirements and regression blocking. Every triage decision records the policy ID that produced it.',
  },
  quality_gate: {
    term: 'Quality gate',
    category: 'Pipeline & statuses',
    short: 'A single pass/fail check of one measurement against one policy threshold.',
    detail: 'Each gate line shows the measured value, the threshold, and the verdict. A label\u2019s triage status is the deterministic result of its gate lines — hover any decision to see them.',
  },
  status_auto_graded: {
    term: 'AUTO_GRADED',
    category: 'Pipeline & statuses',
    short: 'Passed every applicable quality gate; verified without human involvement.',
    detail: 'The label met all policy thresholds (geometry, confidence, consensus, tracking, anomaly). Eligible for training data at the automated confidence tier.',
  },
  status_flagged: {
    term: 'FLAGGED',
    category: 'Pipeline & statuses',
    short: 'Failed one or more gates; routed to human review.',
    detail: 'The failure reasons list which gates failed. Flagged labels enter the HITL queue ordered by severity and safety criticality.',
  },
  status_verified: {
    term: 'VERIFIED',
    category: 'Pipeline & statuses',
    short: 'A human reviewer confirmed (or corrected) this label.',
    detail: 'Highest trust tier. Only VERIFIED labels enter training datasets; corrections flow back as new ground truth.',
  },
  status_rejected: {
    term: 'REJECTED',
    category: 'Pipeline & statuses',
    short: 'A human reviewer judged the label wrong and it was discarded.',
    detail: 'Rejected labels are excluded from training and count against the auto-labeler\u2019s precision in audit reporting.',
  },
  status_pending: {
    term: 'PENDING',
    category: 'Pipeline & statuses',
    short: 'Not yet evaluated — waiting in the pipeline queue.',
    detail: 'No triage decision exists yet for this label.',
  },
  verification_rate: {
    term: 'Verification rate',
    category: 'Pipeline & statuses',
    short: 'Share of processed labels that reached VERIFIED status.',
    detail: 'VERIFIED / all triaged. Rises as HITL clears flagged items and as auto-grading improves.',
  },
  automation_rate: {
    term: 'Automation rate',
    category: 'Pipeline & statuses',
    short: 'Share of labels fully handled without any human touch.',
    detail: 'AUTO_GRADED / all triaged. The core economic metric: every point of automation is human review budget freed for the hard cases.',
  },
  do_not_promote: {
    term: 'DO NOT PROMOTE',
    category: 'Pipeline & statuses',
    short: 'The candidate model failed the promotion policy against the baseline.',
    detail: 'Promotion compares headline and safety metrics with policy-defined maximum allowed drops (e.g. recall −1 pt, safety recall −0.5 pt) and scans all cohorts for localized regressions. Any violated rule is listed with its measured delta and threshold.',
  },
  regression_detected: {
    term: 'REGRESSION',
    category: 'Pipeline & statuses',
    short: 'A cohort where the candidate is significantly worse than the baseline.',
    detail: 'Cohort-level metric delta beyond the policy tolerance with sufficient n. The line shows cohort, metric, baseline → candidate values and the delta.',
  },
  run_status: {
    term: 'Run status',
    category: 'Pipeline & statuses',
    short: 'Lifecycle stage of an evaluation run.',
    detail: 'created → queued → running (workers score partitions) → reducing (merging partial stats) → materializing (writing cube, error index, sketches) → published (queryable) or failed.',
  },
  triage: {
    term: 'Automated triage',
    category: 'Pipeline & statuses',
    short: 'The deterministic decision layer that routes every label to auto-grade, flag or reject.',
    detail: 'Runs each label\u2019s evaluation evidence through the quality policy gates and records status + failure reasons + policy ID. Fully auditable: same evidence + policy ⇒ same decision.',
  },
  process_units: {
    term: 'Process units',
    category: 'Operations',
    short: 'Normalized compute-cost currency consumed by pipeline stages.',
    detail: 'Each stage (labeling, evaluation, anomaly ensembles, training) reports its consumption in a common unit so cost per verified event / per million frames can be tracked and budgeted.',
  },

  // ------------------------------------------------------------ failure reasons
  reason_low_iou: {
    term: 'LOW_IOU',
    category: 'Failure reasons',
    short: 'Box overlaps reference below the policy minimum 3D IoU.',
    detail: 'The label localizes the right object poorly (or the wrong object). See Position/Orientation error to understand which.',
  },
  reason_position_error: {
    term: 'POSITION_ERROR',
    category: 'Failure reasons',
    short: 'Box center too far from the reference position.',
    detail: 'Exceeded max position error (m) in the quality policy.',
  },
  reason_orientation_error: {
    term: 'ORIENTATION_ERROR',
    category: 'Failure reasons',
    short: 'Heading disagrees with reference beyond tolerance.',
    detail: 'Often a 180° flip; corrupts velocity and intent downstream.',
  },
  reason_insufficient_point_support: {
    term: 'INSUFFICIENT_POINT_SUPPORT',
    category: 'Failure reasons',
    short: 'Too few LiDAR points inside the box to verify it.',
    detail: 'Below policy minimum points-in-box. The object may be real (far / occluded) but cannot be auto-verified.',
  },
  reason_sensor_disagreement: {
    term: 'SENSOR_DISAGREEMENT',
    category: 'Failure reasons',
    short: 'Camera and LiDAR evidence conflict for this object.',
    detail: 'One modality sees the object, the other does not (projection mismatch). Routes to human review with both views attached.',
  },
  reason_anomaly: {
    term: 'ANOMALY',
    category: 'Failure reasons',
    short: 'Ensemble anomaly score above the policy threshold.',
    detail: 'The label is statistically unusual vs the population. Could be an error or a genuinely rare event — that is what review decides.',
  },
  reason_grader_disagreement: {
    term: 'GRADER_DISAGREEMENT',
    category: 'Failure reasons',
    short: 'Human graders disagreed below the consensus threshold.',
    detail: 'Routed to senior review; also feeds the grader-pool agreement (Kappa/Alpha) monitoring.',
  },
  reason_id_switch: {
    term: 'ID_SWITCH',
    category: 'Failure reasons',
    short: 'Track identity changed mid-trajectory.',
    detail: 'Hard tracking failure; see Tracking metrics.',
  },
  reason_track_fragmentation: {
    term: 'TRACK_FRAGMENTATION',
    category: 'Failure reasons',
    short: 'Trajectory broken into fragments or track quality below threshold.',
    detail: 'Frequently co-occurs with occlusion and low point density.',
  },
  reason_low_confidence: {
    term: 'LOW_CONFIDENCE',
    category: 'Failure reasons',
    short: 'Model confidence below the policy minimum.',
    detail: 'Not necessarily wrong — low-confidence strata get elevated human sampling instead of automatic rejection.',
  },
  reason_model_regression: {
    term: 'MODEL_REGRESSION',
    category: 'Failure reasons',
    short: 'The producing model version is regressed vs baseline for this slice.',
    detail: 'Labels from a regressed model are blocked from auto-grading when the policy sets block_on_model_regression.',
  },
  // error index types
  error_tp: {
    term: 'TP (True Positive)',
    category: 'Failure reasons',
    short: 'A predicted label correctly matched to a real object.',
    detail: 'Predicted box matched a reference object at 3D IoU ≥ 0.5. The numerator of both precision and recall.',
  },
  error_fn: {
    term: 'FN (False Negative)',
    category: 'Failure reasons',
    short: 'A reference object the model missed entirely.',
    detail: 'No predicted box matched this object at IoU ≥ 0.5. The dominant and most safety-relevant error class.',
  },
  error_fp: {
    term: 'FP (False Positive)',
    category: 'Failure reasons',
    short: 'A predicted box with no corresponding real object.',
    detail: 'Phantom label. Inflates precision problems and can cause ghost braking downstream.',
  },
  error_localization: {
    term: 'LOCALIZATION',
    category: 'Failure reasons',
    short: 'Object matched, but geometry is significantly off.',
    detail: 'IoU passed the match floor but position/orientation/size error breached policy.',
  },
  error_low_conf: {
    term: 'LOW_CONF',
    category: 'Failure reasons',
    short: 'Correct-looking label with confidence below the review threshold.',
    detail: 'Indexed so review sampling can target the uncertain mass.',
  },

  // ------------------------------------------------------------ ground truth
  gt_pseudo: {
    term: 'PSEUDO_GROUND_TRUTH',
    category: 'Ground truth',
    short: 'Reference derived from the auto-labeling consensus itself — not human truth.',
    detail: 'Generated by ensembling model outputs over time (offline tracking, multi-frame smoothing). Enables evaluation at scale but shares blind spots with the models that produced it.',
    caveat: 'Metrics against pseudo-GT are upper bounds: an error both systems make is invisible. Never promote a model on pseudo-GT numbers alone.',
  },
  gt_vendor: {
    term: 'VENDOR_GROUND_TRUTH',
    category: 'Ground truth',
    short: 'Labels purchased from an external annotation vendor.',
    detail: 'Human-drawn but with variable QA depth; treated as medium-confidence reference. Vendor batches are spot-audited via the sampling pipeline.',
  },
  gt_human_verified: {
    term: 'HUMAN_VERIFIED_GROUND_TRUTH',
    category: 'Ground truth',
    short: 'Labels confirmed or corrected by this platform\u2019s own HITL review.',
    detail: 'High-confidence reference: each label carries its reviewer trail and gate evidence.',
  },
  gt_gold: {
    term: 'GOLD_STANDARD',
    category: 'Ground truth',
    short: 'Expert multi-pass annotations — the calibration reference.',
    detail: 'Small, expensive, maximum-quality set used to calibrate every other GT tier and the graders themselves.',
  },
  gt_coverage: {
    term: 'GT coverage',
    category: 'Ground truth',
    short: 'Fraction of the dataset that has any reference ground truth.',
    detail: 'Labels without reference GT can only be checked by GT-free gates (geometry plausibility, sensor consistency, anomaly, consensus).',
  },

  // ------------------------------------------------------------ operations
  sse: {
    term: 'Live stream (SSE)',
    category: 'Operations',
    short: 'This panel updates in real time over a server-sent-events stream.',
    detail: 'The backend pushes pipeline, queue and run-progress state every second on /api/events/stream; no refresh needed.',
  },
  partition: {
    term: 'Partition',
    category: 'Operations',
    short: 'One shard of the population, evaluated independently by a worker.',
    detail: 'Workers emit partial sufficient statistics per partition; the reduce step merges them. Stand-in for distributed executors on this single-node deployment.',
  },
  throughput: {
    term: 'Throughput',
    category: 'Operations',
    short: 'Objects evaluated per second across all active workers.',
    detail: 'Instantaneous rate from completed partitions; drives the ETA estimate.',
  },
  query_cache: {
    term: 'Query cache',
    category: 'Operations',
    short: 'In-memory result cache for aggregate queries.',
    detail: 'Keyed by a hash of (run, filters, group-by, metrics). Hit rate shown in the header; identical dashboard queries return in ~0.1 ms.',
    caveat: 'Invalidated when a run publishes new artifacts.',
  },

  // ------------------------------------------------- root cause analysis (RCA)
  psi: {
    term: 'PSI (Population Stability Index)',
    category: 'Aggregation & scale',
    short: 'Practical magnitude of a distribution change between two populations.',
    detail: 'Sum over bins of (actual% − expected%) × ln(actual%/expected%). Rule of thumb: <0.02 negligible, <0.1 small, <0.25 moderate, ≥0.25 large. Used in the Root Cause Lab to compare offline vs shadow populations independently of p-values.',
    caveat: 'At large n almost any change is statistically significant; PSI answers "does it matter", not "is it detectable".',
  },
  effective_sample_size: {
    term: 'Effective sample size',
    category: 'Human grading & sampling',
    short: 'The number of independent observations your correlated data is actually worth.',
    detail: 'Rows from the same entity (scene/track) are correlated; n_eff = n / design-effect where the design effect grows with cluster size and intra-cluster correlation. Confidence intervals in the Root Cause Lab use n_eff, not raw row counts.',
    caveat: 'Using raw n on clustered evaluation data makes CIs look several times narrower than they really are.',
  },
  simpsons_paradox: {
    term: "Simpson's paradox",
    category: 'Aggregation & scale',
    short: 'Aggregate trend reverses the trend present in every segment.',
    detail: 'If a model improves within every segment but the population mix shifts toward segments where it is weaker, the aggregate can show a regression that no segment shows. The conditional-performance heatmap in the Root Cause Lab surfaces this: consistent per-segment deltas + a mix shift.',
  },
  training_serving_skew: {
    term: 'Training–serving skew',
    category: 'Operations',
    short: 'A feature is computed differently offline than in production serving.',
    detail: 'Unit changes, normalization differences or missing upstream signals make the serving-time feature distribution diverge from the offline one; the model then sees inputs unlike its training data. Detected in the feature-parity stage as a large standardized mean difference concentrated in specific features.',
    caveat: 'Diagnose within segments: an aggregate feature delta can also be an honest population shift.',
  },
  shadow_evaluation: {
    term: 'Shadow evaluation',
    category: 'Operations',
    short: 'Running a candidate model on live traffic without acting on its output.',
    detail: 'The candidate scores real production inputs in parallel with the incumbent; outcomes are compared post-hoc. Sensitive to sampling bias, serving-config differences and provisional labels — which is why the Root Cause Lab audits the shadow pipeline before trusting its number.',
  },

  // ------------------------------------------------- rare-event mining (raremine)
  costumed_pedestrian: {
    term: 'Costumed pedestrian',
    category: 'Anomaly & rarity',
    short: 'A person whose costume distorts the human silhouette a detector relies on.',
    detail: 'Mascot / inflatable / animal / character / robot-armor / oversized / large-prop costumes break silhouette, texture and proportion priors, causing misses or misclassification (vehicle, animal, background). The Rare-Event Miner keeps three SEPARATE confidences: a human is present, a costume is present, and the combination is a rare event.',
    caveat: 'Confounders (mascot statues, inflatable decorations, printed figures, mannequins) look identical in one frame — only motion, 3D volume, or human validation separates them.',
  },
  leakage_guard: {
    term: 'Leakage guard',
    category: 'Operations',
    short: 'Examples in protected evaluation sets can never silently become training data.',
    detail: 'Anything routed to a REGRESSION or SAFETY_CRITICAL evaluation set gets training_eligible=false in its lineage. Promoting such an example to training requires an explicit governance override recording who and why, which is permanently audited.',
  },
  mining_calibration: {
    term: 'Mining calibration',
    category: 'Anomaly & rarity',
    short: 'Whether the miner\u2019s stated confidence matches how often it is right.',
    detail: 'Candidates are binned by stated rare-event confidence; within each bin, the observed rate of true (planted) rare events is measured. A calibrated miner\u2019s 80% bin is right about 80% of the time — over-confident bins mean stated confidence cannot be trusted for triage ordering.',
  },

  // ------------------------------------------------------------ hardware acceleration (Vitis)
  vitis_emulated_backend: {
    term: 'Emulated Vitis backend',
    category: 'Hardware acceleration',
    short: 'CPU emulator of a Vitis Vision FPGA pipeline: same ops, faithful hardware constraints, no silicon.',
    detail: 'Implements the same VisionBackend interface as the reference float32 backend, but applies ap_fixed<W,I> fixed-point quantization (truncation + saturation), XFCVDEPTH line-buffer limits and LUT divide/sqrt to every op, plus a deterministic per-op latency model (pixels/cycle × clock, PL vs AIE placement). A real Vitis/PYNQ/XRT backend slots in behind the identical interface later.',
    caveat: 'No FPGA hardware is attached to this machine. Every latency, throughput and speedup figure this backend reports is analytically MODELED, never measured on silicon.',
  },
  ap_fixed_quantization: {
    term: 'ap_fixed<W,I> quantization',
    category: 'Hardware acceleration',
    short: 'HLS fixed-point format: W total bits, I integer bits; values are truncated and saturated like on the FPGA.',
    detail: 'Matches the HLS ap_fixed defaults: rounding is truncation toward negative infinity (AP_TRN) at 2^-(W-I) resolution, and overflow saturates (AP_SAT) to the representable range. Fewer fractional bits mean coarser detection confidences and positions — the root cause the HIL ablation isolates as "precision".',
  },
  xfcvdepth: {
    term: 'XFCVDEPTH (line-buffer depth)',
    category: 'Hardware acceleration',
    short: 'Maximum image width a streaming Vitis Vision kernel can buffer rows for.',
    detail: 'Streaming FPGA kernels hold a few rows in on-chip line buffers whose width is bounded at synthesis time. When the frame is wider than the configured depth, the emulator processes independent vertical strips with no halo exchange — producing localized seam artifacts at strip boundaries, exactly like an under-provisioned kernel.',
  },
  lut_approximation: {
    term: 'LUT divide/sqrt',
    category: 'Hardware acceleration',
    short: 'HLS-style lookup-table approximation of divide and square root with bounded relative error.',
    detail: 'Full dividers are expensive in fabric, so HLS designs use mantissa-indexed lookup tables. The emulator uses 2^lut_bits-entry tables (relative error ≈ 2^-lut_bits) for every reciprocal and square root — including the determinant inverse inside the optical-flow solver.',
  },
  quantization_gap_score: {
    term: 'Quantization gap score',
    category: 'Hardware acceleration',
    short: 'Scalar severity of the float32-vs-fixed-point detection gap on identical frames.',
    detail: 'Computed from paired per-object comparisons: 0.45·dropped-detection rate + 0.2·class-flip rate + 0.2·mean |confidence drift| + 0.15·(1 − mean pair IoU). The HIL ablation re-measures it with one hardware constraint enabled at a time to attribute the gap to precision, streaming depth, or LUT approximation.',
  },
  modeled_throughput: {
    term: 'Modeled FPGA throughput',
    category: 'Hardware acceleration',
    short: 'Analytical latency estimate (pixels/cycle × clock MHz), clearly labeled — never a measurement.',
    detail: 'Each op has a modeled pixels/cycle rate and PL-vs-AIE placement; end-to-end pipelined fps assumes HLS dataflow overlap (1 / max stage latency). Reported alongside the MEASURED CPU wall time of the reference pipeline for contrast.',
    caveat: 'Ignores memory bandwidth, AXI backpressure and resource contention; it is an upper bound to be validated on silicon, and every report carries modeled_not_measured: true.',
  },
  flow_motion_baseline: {
    term: 'Flow motion baseline',
    category: 'Hardware acceleration',
    short: 'Model-independent object motion predicted by dense pyramidal Lucas-Kanade optical flow.',
    detail: 'Flow over consecutive BEV rasters predicts each ground-truth object\u2019s next position without consulting any detection engine. A track is "flow-continuous" at a frame when the flow-predicted displacement matches the true displacement within a residual gate — the license to blame the engine (not the scene) for a dropped detection.',
  },
  temporal_stability_score: {
    term: 'Temporal stability score',
    category: 'Hardware acceleration',
    short: 'Per-engine 0–100 score penalizing flicker, jitter, fragmentation and unexcused ID switches against the flow baseline.',
    detail: '100 × (0.35·(1 − 4·flicker rate) + 0.25·e^(−1.5·jitter) + 0.25·(1 − 0.6·fragmentation/track) + 0.15·(1 − unexcused ID-switch fraction)). Computed per cohort (day/night/rain/occluded) and per backend; the backend-agreement meta-check verifies fixed-point flow does not change the engine ranking.',
    caveat: 'The weights are a documented product decision; check ranking sensitivity before gating releases on the absolute score.',
  },
  evaluation_only_variant: {
    term: 'Evaluation-only variant',
    category: 'Hardware acceleration',
    short: 'Synthetic stress variant barred from training by default: full lineage, protected eval destination.',
    detail: 'Every generated augmentation variant carries training_eligible: false, evaluation_only: true and destination REGRESSION_EVALUATION_SET, plus complete lineage (recipe with resolved parameters, seed, source frame, backend config). Mirrors the rare-event miner\u2019s protected-destination leakage guard: stress data that leaked into training would corrupt the very regression signal it exists to protect.',
  },

  // ------------------------------------------------------------ EM readiness (Hill Climbing EM)
  hc_competency: {
    term: 'Competency',
    category: 'EM readiness',
    short: 'A single skill node in the 4-phase blueprint graph, with prerequisites and a dimension tag.',
    detail: 'Each competency belongs to a phase (ML depth / system design / execution & people / simulation), lists prerequisite competencies, and is tagged with one dimension (Knowledge, Technical Reasoning, Leadership, Execution). Scores are tracked per competency and per dimension — never collapsed into one number.',
  },
  hc_readiness_state: {
    term: 'Readiness state',
    category: 'EM readiness',
    short: 'Per-competency progression: NOT_STARTED → LEARNING → PRACTICING → NEEDS_REVIEW → COMPETENT → STRONG → INTERVIEW_READY.',
    detail: 'Derived from three separate scores: knowledge (diagnostic/interview answers), application (exercises, design lab, simulation) and evidence (STAR stories and other artifacts). NEEDS_REVIEW is entered when recent attempts contradict earlier strong scores.',
    caveat: 'States are only as trustworthy as the evidence backing them; with the LLM offline, scoring is rule-based concept coverage, which is stricter but coarser.',
  },
  hc_evidence_artifact: {
    term: 'Evidence artifact',
    category: 'EM readiness',
    short: 'A stored, quotable record (attempt, STAR story, design grade, simulation debrief, interview transcript) that justifies a score.',
    detail: 'The scoring rule is "no score without evidence": every evaluation must quote specific user statements. Artifacts persist under runs/hillclimb/ and are browsable in the Evidence Library; clicking any matrix score shows the artifacts behind it.',
  },
  hc_bottleneck: {
    term: 'Bottleneck competency',
    category: 'EM readiness',
    short: 'The weak prerequisite blocking the most downstream competencies — not merely the lowest score.',
    detail: 'Computed from the prerequisite graph: for each weak competency, count the downstream competencies transitively gated on it; the highest-leverage weakness wins. Fixing the bottleneck unblocks more of the graph than fixing the globally lowest score.',
  },
  hc_next_best_action: {
    term: 'Next best action',
    category: 'EM readiness',
    short: 'Exactly one concept to study, one exercise to attempt and one assessment to take, aimed at the current bottleneck.',
    detail: 'Regenerated whenever the readiness matrix changes. Deliberately singular: a ranked to-do list invites cherry-picking easy items; one action per category forces work on the highest-leverage weakness.',
  },
  hc_claim_vs_evidence: {
    term: 'Claim vs evidence',
    category: 'EM readiness',
    short: 'STAR Story Box flag: an unquantified claim ("improved performance") vs measurable evidence (numbers, before/after, named mechanism).',
    detail: 'Claim sentences are detected by improvement verbs without attached quantities. Each flag asks for the missing strengthening: the metric, the baseline, the delta, or the mechanism that caused it. Interviewers apply the same test — so the diagnoser applies it first.',
  },
  hc_anti_gaming: {
    term: 'Anti-gaming rule',
    category: 'EM readiness',
    short: 'Verbosity alone must not raise scores: concept coverage, tradeoffs and quantified results are scored, not length.',
    detail: 'The evaluator scores rubric-concept coverage, tradeoff discussion and quantified statements. Length is not a feature, and low information density is penalized — a long waffle scores below a short precise answer (enforced by test).',
  },
  hc_hill_climbing: {
    term: 'Hill climbing (simulation)',
    category: 'EM readiness',
    short: 'Iterative optimization loop: hypothesis → intervention → measure → keep/reject → repeat, under competing objectives.',
    detail: 'The Phase-4 simulation tracks ten competing state metrics (performance, safety, reliability, cost, velocity, maintainability, morale, customer impact, risk, schedule) with hard floors — dropping safety below its floor triggers an incident. The balanced multi-objective score rewards keeping all objectives healthy, not maximizing one.',
    caveat: 'Deterministic given the scenario seed: identical intervention sequences replay identically, which is what makes debriefs auditable.',
  },
};

export type GlossaryKey = keyof typeof GLOSSARY;

/** Resolve a status / reason string (e.g. "FLAGGED", "LOW_IOU") to a glossary key, if known. */
export function glossaryKeyForStatus(status: string): string | null {
  const s = status.toLowerCase();
  const direct: Record<string, string> = {
    auto_graded: 'status_auto_graded',
    flagged: 'status_flagged',
    verified: 'status_verified',
    rejected: 'status_rejected',
    pending: 'status_pending',
    low_iou: 'reason_low_iou',
    position_error: 'reason_position_error',
    orientation_error: 'reason_orientation_error',
    insufficient_point_support: 'reason_insufficient_point_support',
    sensor_disagreement: 'reason_sensor_disagreement',
    anomaly: 'reason_anomaly',
    grader_disagreement: 'reason_grader_disagreement',
    id_switch: 'reason_id_switch',
    track_fragmentation: 'reason_track_fragmentation',
    low_confidence: 'reason_low_confidence',
    model_regression: 'reason_model_regression',
    tp: 'error_tp',
    fn: 'error_fn',
    fp: 'error_fp',
    localization: 'error_localization',
    low_conf: 'error_low_conf',
    pseudo_ground_truth: 'gt_pseudo',
    vendor_ground_truth: 'gt_vendor',
    human_verified_ground_truth: 'gt_human_verified',
    gold_standard: 'gt_gold',
  };
  return direct[s] ?? null;
}

/** Glossary keys for gate ids as they appear in gate lines. */
export const GATE_GLOSSARY: Record<string, string> = {
  iou_3d: 'iou_3d',
  position_error_m: 'position_error',
  orientation_error_deg: 'orientation_error',
  has_reference_gt: 'gt_coverage',
  points_in_box: 'points_in_box',
  box_occupancy: 'point_in_box_ratio',
  centroid_consistency_m: 'position_error',
  ground_contact_error_m: 'ground_contact',
  box_dimensions_vs_class_prior: 'dimension_error',
  dimension_error: 'dimension_error',
  sensor_consistency_cam_lidar: 'sensor_consistency',
  sensor_consistency: 'sensor_consistency',
  geometric_validation: 'quality_gate',
  anomaly_score: 'anomaly_score',
  grader_consensus: 'grader_consensus',
  id_switch: 'id_switch',
  track_fragmentation: 'fragmentation',
  track_quality: 'track_quality',
  detection_confidence: 'confidence',
  model_regression: 'reason_model_regression',
};

export const GLOSSARY_CATEGORIES: GlossaryCategory[] = [
  'Detection metrics',
  'Geometry & LiDAR',
  'Tracking',
  'Anomaly & rarity',
  'Safety (SSAM)',
  'Human grading & sampling',
  'Aggregation & scale',
  'Pipeline & statuses',
  'Failure reasons',
  'Ground truth',
  'Operations',
  'EM readiness',
  'Hardware acceleration',
];
