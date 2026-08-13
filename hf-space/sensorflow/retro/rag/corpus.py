"""Seed safety-case corpus.

EVERYTHING HERE IS SYNTHETIC DEMONSTRATION MATERIAL. Every synthetic rule is
tagged SYNTHETIC_EXAMPLE / NOT_A_REAL_STANDARD in the text itself AND in the
chunk metadata, so no rendering path can strip the label. The SOTIF document
is an honest concept-level paraphrase of ISO 21448 ideas, explicitly labeled
PARAPHRASE_NOT_STANDARD_TEXT — it contains no standard text and no clause
numbers, and must never be cited as the standard.
"""

from __future__ import annotations

from typing import Dict, List

SYNTH = "SYNTHETIC_EXAMPLE / NOT_A_REAL_STANDARD"

SEED_DOCUMENTS: List[Dict] = [
    {
        "doc_id": "SFS-SAFE-001",
        "document": "Internal Perception Safety Requirements",
        "source": "sensorflow-internal",
        "version": "2.1",
        "jurisdiction": "internal",
        "doc_type": "safety_requirement",
        "effective_date": "2026-01-15",
        "synthetic": "true",
        "label": SYNTH,
        "text": f"""[{SYNTH}] Internal Perception Safety Requirements v2.1.

## VRU detection within stopping distance
[{SYNTH}] REQ-VRU-01: A vulnerable road user (pedestrian, cyclist,
motorcyclist) that is inside the ego vehicle's current stopping distance
envelope MUST be detected and tracked with safety-critical recall of at
least 0.999 per exposure hour. A false negative (missed detection) of a
pedestrian within stopping distance is classified at minimum CRITICAL and
requires a retrospective before any launch decision. Degraded weather (rain,
fog, spray) does not relax this requirement; it tightens the sensor fusion
redundancy obligations in REQ-VRU-03.

## False positive braking limits
[{SYNTH}] REQ-FP-02: Phantom braking caused by misclassified static or
low-mass clutter (plastic bag, cardboard, tire shred, road debris) MUST NOT
produce a hard brake exceeding 0.35 g when no collision-relevant object is
present. A phantom hard brake above this magnitude with following traffic
present is classified at minimum DISRUPTIVE, and CRITICAL when rear
time-gap is under 1.5 seconds, because induced rear-end collision risk
transfers harm to other road users.

## Detection latency budget
[{SYNTH}] REQ-LAT-04: End-to-end perception-to-planner reaction latency
(sensor timestamp to planner action) MUST NOT exceed 300 ms at P99 for
objects inside the safety-critical zone. Latency consumed by perception
reduces the remaining human or system reaction time budget and is charged
against stopping distance in the safety case.

## Sensor fusion disagreement
[{SYNTH}] REQ-VRU-03: When camera and LiDAR disagree on the existence of a
VRU-class object inside 2x stopping distance, the stack MUST treat the
object as present (existence-favoring fusion) until disproven by two
consecutive frames of agreement.""",
    },
    {
        "doc_id": "SFS-LAUNCH-004",
        "document": "Launch Criteria and Release Gate Policy",
        "source": "sensorflow-internal",
        "version": "3.0",
        "jurisdiction": "internal",
        "doc_type": "launch_criteria",
        "effective_date": "2026-03-01",
        "synthetic": "true",
        "label": SYNTH,
        "text": f"""[{SYNTH}] Launch Criteria and Release Gate Policy v3.0.

## Launch gate conditions
[{SYNTH}] GATE-01: A candidate model MUST NOT receive a PASS launch
recommendation while any retrospective scorecard carries an unresolved
severity of CRITICAL or FATAL. GATE-02: Safety-critical recall (SCR) on the
criticality-weighted evaluation population must not regress versus baseline
beyond the policy tolerance of 0.2 percentage points with statistical
significance. GATE-03: A determination of INSUFFICIENT_EVIDENCE can never be
converted into PASS by any agent, human or automated; the only allowed
transitions are to CONDITIONAL_PASS after new evidence or to FAIL.

## Human review triggers
[{SYNTH}] GATE-05: Human review is mandatory when (a) severity is CRITICAL
or FATAL, (b) the AI-proposed severity diverges from the deterministic
policy severity, (c) any evidence field required by the scorecard is
UNKNOWN, or (d) the analysis relied on fewer than two independent evidence
sources.

## Disengagement and collision-risk events
[{SYNTH}] GATE-07: Any event with a collision-risk trajectory (predicted
time-to-collision below 1.5 s against a VRU) or an actual disengagement
attributed to perception MUST be retrospectively analyzed before the next
release train, with the scorecard archived and auditable.""",
    },
    {
        "doc_id": "SFS-PERC-002",
        "document": "Perception Requirements in Degraded Conditions",
        "source": "sensorflow-internal",
        "version": "1.4",
        "jurisdiction": "internal",
        "doc_type": "perception_requirement",
        "effective_date": "2025-11-20",
        "synthetic": "true",
        "label": SYNTH,
        "text": f"""[{SYNTH}] Perception Requirements in Degraded Conditions v1.4.

## Rain and spray degradation
[{SYNTH}] REQ-WX-01: In measured rain rates up to 25 mm/h, pedestrian
detection recall inside the safety-critical zone must degrade by no more
than 0.5 percentage points versus clear-weather baseline. LiDAR point
density loss and camera contrast loss must be compensated by fusion
weighting, not by raising confidence thresholds, because threshold raising
silently trades false positives for missed VRUs.

## Confidence calibration
[{SYNTH}] REQ-CAL-02: Detection confidence must be calibrated such that a
reported confidence of 0.5 corresponds to an empirical precision between
0.45 and 0.55 on the rolling evaluation window. Systematic underconfidence
in rain is a known failure mode and must be tracked as a calibration defect,
not compensated with global threshold changes.

## Object class stability
[{SYNTH}] REQ-CLS-03: Class flicker (an object toggling between classes such
as vehicle and unknown or debris across consecutive frames) inside 2x
stopping distance must trigger existence-favoring handling and is reportable
evidence in any retrospective involving misclassification.""",
    },
    {
        "doc_id": "SFS-ODD-003",
        "document": "Operational Design Domain Definitions",
        "source": "sensorflow-internal",
        "version": "2.0",
        "jurisdiction": "internal",
        "doc_type": "odd_definition",
        "effective_date": "2025-09-01",
        "synthetic": "true",
        "label": SYNTH,
        "text": f"""[{SYNTH}] Operational Design Domain Definitions v2.0.

## Urban ODD envelope
[{SYNTH}] ODD-URB-01: The urban ODD covers divided and undivided surface
streets with posted speeds up to 45 mph (20.1 m/s), signalized and
unsignalized intersections, and marked crosswalks. Ego speed above the
posted limit voids the ODD envelope for analysis purposes.

## Weather boundaries
[{SYNTH}] ODD-WX-02: Operation is inside ODD in light-to-moderate rain
(under 25 mm/h) with functional wipers and defog. Heavy rain above 25 mm/h,
standing water, and night-plus-heavy-rain combinations are OUTSIDE the
current ODD; events occurring there are ODD-exit events, and the correct
system response is a minimal-risk maneuver, not continued operation.

## Analysis obligations
[{SYNTH}] ODD-AN-03: Every retrospective must state whether the event was
inside or outside ODD, because requirements applicability, severity
weighting, and launch gating differ. Outside-ODD events still require
analysis of whether the ODD-exit detection itself functioned.""",
    },
    {
        "doc_id": "SFS-RETRO-HIST-01",
        "document": "Historical Retrospectives Digest",
        "source": "sensorflow-internal",
        "version": "1.0",
        "jurisdiction": "internal",
        "doc_type": "historical_retrospective",
        "effective_date": "2025-12-10",
        "synthetic": "true",
        "label": SYNTH,
        "text": f"""[{SYNTH}] Historical Retrospectives Digest v1.0.

## RETRO-2025-014 phantom braking cluster
[{SYNTH}] Twelve phantom hard-brake events in one quarter were traced to
low-mass road debris (plastic bags, mylar balloons) being classified as
vehicle or unknown-obstacle with high confidence at 30-60 m range. Root
cause hypothesis confirmed: training distribution lacked wind-blown
deformable objects; the tracker latched a spurious velocity toward ego from
bag motion, producing a false closing-velocity estimate and an unwarranted
emergency braking response. Fix: hard-negative mining of deformable clutter
plus track-consistency gating on mass-plausibility. Post-fix phantom brake
rate fell 87 percent.

## RETRO-2025-021 missed pedestrian in rain
[{SYNTH}] A pedestrian in dark clothing crossing mid-block in moderate rain
at night was detected 1.9 s later than required, inside stopping distance.
Camera contrast was degraded and LiDAR returns were attenuated by spray;
fusion confidence stayed below the tracking threshold for 6 consecutive
frames. The planner received the object late and executed a hard brake with
residual collision risk. Contributing factors: confidence threshold raised
globally two releases earlier to suppress rain clutter false positives —
exactly the trade REQ-WX-01 prohibits. Fix: rain-conditioned fusion
weighting and restoration of threshold, plus targeted rain-night pedestrian
data collection.""",
    },
    {
        "doc_id": "SFS-EVAL-005",
        "document": "Evaluation Policies for Safety-Relevant Metrics",
        "source": "sensorflow-internal",
        "version": "1.2",
        "jurisdiction": "internal",
        "doc_type": "evaluation_policy",
        "effective_date": "2026-02-05",
        "synthetic": "true",
        "label": SYNTH,
        "text": f"""[{SYNTH}] Evaluation Policies for Safety-Relevant Metrics v1.2.

## Safety-critical recall definition
[{SYNTH}] EVAL-SCR-01: Safety-critical recall (SCR) is recall computed only
over ground-truth objects whose criticality context meets the policy
criteria: object inside 1.5x stopping distance, closing relative velocity,
and VRU class or collision-relevant mass. SCR is reported with its
denominator; an SCR delta without a criticality-context denominator is not
admissible launch evidence.

## Asymmetric error costs
[{SYNTH}] EVAL-COST-02: False negatives and false positives carry
asymmetric, CONTEXT-DEPENDENT costs. A missed distant non-closing object can
be BENIGN while a phantom hard brake with a tailgater can be CRITICAL. Cost
is computed from context (class, distance versus stopping distance, relative
motion, remaining reaction time, intervention magnitude), never from the
error type alone.

## Statistical significance
[{SYNTH}] EVAL-SIG-03: Metric deltas feeding launch decisions must carry
anytime-valid confidence sequences (the platform's seqeval machinery) or an
explicit NOT_SIGNIFICANT marker. Point deltas without uncertainty are
inadmissible.""",
    },
    {
        "doc_id": "SOTIF-CONCEPTS-21448",
        "document": "SOTIF (ISO 21448) Concept Summary",
        "source": "public-standard-paraphrase",
        "version": "concept-summary-1.0",
        "jurisdiction": "international",
        "doc_type": "standard_concept_paraphrase",
        "effective_date": "2022-06-01",
        "synthetic": "false",
        "label": "PARAPHRASE_NOT_STANDARD_TEXT",
        "text": """[PARAPHRASE_NOT_STANDARD_TEXT] This is an honest concept-level
paraphrase of ideas associated with ISO 21448 (Safety of the Intended
Functionality, SOTIF). It contains NO standard text and NO clause numbers
and must never be cited as the standard itself.

## Functional insufficiency
[PARAPHRASE_NOT_STANDARD_TEXT] SOTIF addresses hazards that arise not from
component faults but from performance limitations of the intended
functionality — for example a perception stack whose sensor physics or
training distribution is insufficient for a scenario (heavy rain, unusual
object appearance). A missed pedestrian in rain with no hardware fault is a
canonical SOTIF-style functional insufficiency, not a malfunction.

## Triggering conditions and scenario space
[PARAPHRASE_NOT_STANDARD_TEXT] The concept framework partitions scenarios
into known-safe, known-hazardous, and unknown-hazardous. Engineering effort
aims to discover unknown-hazardous scenarios (triggering conditions such as
specific weather, lighting, or object configurations) and move them into the
known space, then reduce their residual risk through requirements, data,
or ODD restriction.

## Verification and validation implication
[PARAPHRASE_NOT_STANDARD_TEXT] Retrospective analysis of field failures is
one of the accepted mechanisms for discovering triggering conditions.
Evidence from such analysis feeds requirement updates and validation
targets; residual risk acceptance must be argued, not assumed.""",
    },
]


def corpus_metadata(doc: Dict) -> Dict[str, str]:
    return {
        "doc_id": doc["doc_id"],
        "document": doc["document"],
        "source": doc["source"],
        "version": doc["version"],
        "jurisdiction": doc["jurisdiction"],
        "doc_type": doc["doc_type"],
        "effective_date": doc["effective_date"],
        "synthetic": doc["synthetic"],
        "label": doc["label"],
    }
