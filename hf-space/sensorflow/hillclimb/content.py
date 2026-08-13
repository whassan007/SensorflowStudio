"""Exercise/content generation: parameterized, seeded scenario templates.

Every competency can generate exercises of the required shape:
{competency_id, difficulty, prerequisites, scenario, expected_reasoning,
 evaluation_rubric, common_failure_modes, follow_up_questions}

Retry with a different seed produces a STRUCTURALLY DIFFERENT problem for the
same competency (different template variant and/or slot values) so users can't
memorize by repetition. When an LLM is available it may enrich the scenario
text, but the rubric/reasoning skeleton is always the deterministic template
output (the rubric is what evaluation trusts).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.hillclimb import llm
from sensorflow.hillclimb.blueprint import competency_index, load_blueprint
from sensorflow.hillclimb.models import Store, get_store, new_id


class RubricItem(BaseModel):
    criterion: str
    # check: "keywords" (concept coverage), "quantified" (numeric evidence),
    # "tradeoff" (explicit tradeoff language). Non-keyword checks are scored
    # structurally by the evaluator.
    check: str = "keywords"
    keywords: List[str] = Field(default_factory=list)
    weight: float = 1.0


class Exercise(BaseModel):
    exercise_id: str = Field(default_factory=lambda: new_id("ex"))
    competency_id: str
    difficulty: int = 2  # 1=warmup 2=core 3=stretch
    prerequisites: List[str] = Field(default_factory=list)
    scenario: str
    expected_reasoning: List[str] = Field(default_factory=list)
    evaluation_rubric: List[RubricItem] = Field(default_factory=list)
    common_failure_modes: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    template_id: str = "generic"
    seed: int = 0
    family: str = "generic"
    linked_tool: Optional[Dict] = None


QUANT_RUBRIC = RubricItem(criterion="Quantifies claims with numbers, thresholds, or before/after comparisons",
                          check="quantified", weight=1.0)
TRADEOFF_RUBRIC = RubricItem(criterion="States explicit tradeoffs and what is sacrificed by the chosen approach",
                             check="tradeoff", weight=1.0)


# ------------------------------------------------------------- template bank


def _offline_shadow(rng: random.Random, difficulty: int) -> Dict:
    """Canonical Phase-1 family: offline metric up, shadow deployment down."""
    offline_gain = rng.choice([3, 4, 5, 6, 7])
    shadow_drop = rng.choice([1, 2, 3, 4])
    metric = rng.choice(["mAP@0.5", "pedestrian recall", "3D IoU pass-rate", "F1 on cyclists"])
    model = rng.choice(["perception-v12", "detector-nightly", "fusion-rc3", "tracker-v8"])
    context = rng.choice([
        "after a training-data refresh that added 40k night-time frames",
        "after switching the loss to a focal variant",
        "after a new auto-labeling vendor batch landed",
        "with no intentional model change (dependency bump only)",
    ])
    variant = rng.choice([
        (f"Your team ships {model}. Offline evaluation shows {metric} improved by +{offline_gain}% "
         f"{context}. But in a one-week shadow deployment against live traffic, the same metric "
         f"is DOWN {shadow_drop}% versus the current production model. Leadership asks whether to ship. "
         f"Diagnose the divergence: enumerate the possible causes, how you would distinguish them, "
         f"and what you would do before making a ship/no-ship call."),
        (f"A staff engineer reports: '{model} gained +{offline_gain}% {metric} offline {context}, "
         f"but shadow-mode monitoring shows a {shadow_drop}% regression.' The launch review is tomorrow. "
         f"Walk through your differential diagnosis of the offline/online gap and the concrete "
         f"experiments that would confirm or eliminate each hypothesis."),
    ])
    return {
        "scenario": variant,
        "expected_reasoning": [
            "Metric definition / computation mismatch between offline eval and shadow monitoring",
            "Distribution shift between the offline eval set and live traffic",
            "Sampling bias in how the offline set (or the shadow sample) was constructed",
            "Feature / training-serving skew in the input pipeline",
            "Serving-stack mismatch (preprocessing, quantization, version drift)",
            "Statistical noise — is the shadow delta even significant?",
            "A true regression the offline set cannot see",
            "A verification plan: slice analysis, matched-sample re-scoring, significance testing",
        ],
        "rubric": [
            RubricItem(criterion="Considers metric definition / computation mismatch",
                       keywords=["metric definition", "computed differently", "matching threshold",
                                 "iou threshold", "different metric", "definition mismatch", "metric error",
                                 "same metric", "measurement mismatch"]),
            RubricItem(criterion="Considers distribution shift between eval set and live traffic",
                       keywords=["distribution shift", "domain shift", "drift", "live traffic differs",
                                 "population", "distribution change", "data shift"]),
            RubricItem(criterion="Considers sampling bias in eval or shadow sample",
                       keywords=["sampling bias", "biased sample", "not representative", "selection bias",
                                 "sample size", "how the eval set was built", "oversampl", "undersampl"]),
            RubricItem(criterion="Considers feature / training-serving skew",
                       keywords=["feature skew", "training-serving skew", "training serving skew",
                                 "feature pipeline", "feature mismatch", "stale feature"]),
            RubricItem(criterion="Considers serving-stack mismatch (preprocessing/quantization/versions)",
                       keywords=["serving", "preprocessing", "quantization", "version mismatch",
                                 "deployment stack", "runtime", "inference stack", "model artifact"]),
            RubricItem(criterion="Questions statistical significance / noise before reacting",
                       keywords=["noise", "significan", "confidence interval", "variance", "sample size",
                                 "error bars", "statistical"]),
            RubricItem(criterion="Keeps 'true regression' on the table and says how to confirm it",
                       keywords=["true regression", "real regression", "genuine regression",
                                 "actually worse", "genuinely regressed"]),
            RubricItem(criterion="Proposes a concrete verification plan (slices, matched re-scoring, A/B)",
                       keywords=["slice", "re-run", "rerun", "a/b", "matched", "re-score", "rescore",
                                 "experiment", "holdout", "ablat", "bisect"]),
            QUANT_RUBRIC, TRADEOFF_RUBRIC,
        ],
        "failure_modes": [
            "Jumping straight to 'the model is worse' without checking the measurement itself",
            "Ignoring statistical significance of the shadow delta",
            "Treating offline eval as ground truth for live behavior",
            "No concrete verification experiment — only speculation",
        ],
        "follow_ups": [
            "Your offline set is auto-labeled. How does that change your diagnosis ordering?",
            f"The {shadow_drop}% shadow drop is concentrated in night-time frames. What does that tell you?",
            "You have 48 hours before the launch review. Which single experiment do you run and why?",
            "How would you redesign the evaluation pipeline so this class of divergence is caught automatically?",
        ],
        "linked_tool": {
            "label": "Practice this diagnosis with the live RCA workbench",
            "page": "rca",
            "api": "/api/rca",
        },
    }


def _cusum_drift(rng: random.Random, difficulty: int) -> Dict:
    window = rng.choice([7, 14, 30])
    magnitude = rng.choice([0.5, 1.0, 2.0])
    signal = rng.choice(["false-positive rate on static objects", "track fragmentation rate",
                         "mean confidence on pedestrians", "per-frame latency p99"])
    return {
        "scenario": (f"Production monitoring shows {signal} creeping up roughly {magnitude}% over the last "
                     f"{window} days, but every daily comparison against the previous day looks 'within noise'. "
                     f"Design a sequential change-detection approach (e.g. CUSUM) for this signal: how it works, "
                     f"how you set thresholds, and how you avoid alert fatigue."),
        "expected_reasoning": [
            "Why daily pairwise comparisons miss slow drift; cumulative evidence",
            "CUSUM mechanics: cumulative sum of deviations from a reference mean, decision interval h, slack k",
            "Threshold selection as a tradeoff between detection delay and false-alarm rate",
            "Baseline/reference maintenance and reset policy after confirmed changes",
            "Operational design: who gets paged, runbooks, alert budgets",
        ],
        "rubric": [
            RubricItem(criterion="Explains why point-in-time comparisons miss slow drift",
                       keywords=["slow drift", "cumulative", "accumulate", "small shifts", "gradual"]),
            RubricItem(criterion="Describes CUSUM mechanics (reference mean, slack, decision interval)",
                       keywords=["cusum", "cumulative sum", "reference mean", "slack", "decision interval",
                                 "control chart", "change point"]),
            RubricItem(criterion="Treats threshold selection as detection-delay vs false-alarm tradeoff",
                       keywords=["false alarm", "detection delay", "threshold", "arl", "sensitivity"]),
            RubricItem(criterion="Covers baseline maintenance / reset after confirmed change",
                       keywords=["baseline", "reset", "re-baseline", "reference update", "recalibrat"]),
            RubricItem(criterion="Addresses alert fatigue and operational response",
                       keywords=["alert fatigue", "pager", "runbook", "on-call", "alert budget", "triage"]),
            QUANT_RUBRIC, TRADEOFF_RUBRIC,
        ],
        "failure_modes": [
            "Proposing a fixed threshold on the raw metric (misses slow drift entirely)",
            "No false-alarm analysis",
            "No plan for what happens after an alert fires",
        ],
        "follow_ups": [
            "The signal is seasonal (weekday/weekend). How do you adapt the detector?",
            "How would you validate the detector before trusting it in production?",
        ],
        "linked_tool": None,
    }


def _parallel_inference(rng: random.Random, difficulty: int) -> Dict:
    qps = rng.choice([2000, 5000, 20000, 50000])
    latency = rng.choice([50, 100, 200])
    model_size = rng.choice(["300M-param detector", "1.2B-param fusion model", "ensemble of 3 detectors"])
    return {
        "scenario": (f"Design the serving layer for a {model_size} that must handle {qps} inferences/sec "
                     f"with a p99 latency budget of {latency}ms. GPUs are expensive and the fleet must survive "
                     f"a zone outage. Walk through batching, sharding, autoscaling, and the capacity math."),
        "expected_reasoning": [
            "Dynamic batching and its latency/throughput tradeoff",
            "Model/tensor sharding vs replica scaling",
            "Capacity math: per-GPU throughput → replica count → headroom",
            "Tail latency: queueing, hedged requests, timeouts, degradation modes",
            "Zone-redundant deployment and the cost of redundancy",
        ],
        "rubric": [
            RubricItem(criterion="Uses dynamic batching with an explicit latency cost",
                       keywords=["batch", "batching", "batch size", "queue delay"]),
            RubricItem(criterion="Distinguishes replica scaling from model sharding",
                       keywords=["shard", "replica", "tensor parallel", "scale out", "horizontal"]),
            RubricItem(criterion="Shows capacity math (per-GPU throughput → fleet size → headroom)",
                       keywords=["capacity", "headroom", "per-gpu", "throughput per", "utilization",
                                 "provision"]),
            RubricItem(criterion="Handles tail latency (queueing, hedging, timeouts, sheddable load)",
                       keywords=["p99", "tail", "hedg", "timeout", "queue", "shed", "degrade"]),
            RubricItem(criterion="Designs for zone failure with explicit redundancy cost",
                       keywords=["zone", "redundan", "failover", "outage", "multi-region", "n+1"]),
            QUANT_RUBRIC, TRADEOFF_RUBRIC,
        ],
        "failure_modes": [
            "No numbers: hand-wavy 'we autoscale' without capacity math",
            "Ignoring tail latency, quoting only mean latency",
            "Redundancy without acknowledging its cost",
        ],
        "follow_ups": [
            "Traffic doubles for 10 minutes every day at rush hour. Buy capacity or shed load — argue it.",
            "The model team wants to ship a 2x bigger model next quarter. What breaks first?",
        ],
        "linked_tool": None,
    }


def _leadership_scenario(rng: random.Random, difficulty: int, comp_name: str, topics: List[str]) -> Dict:
    tension = rng.choice([
        "your strongest senior engineer openly disagrees with the plan in team channels",
        "the PM has promised the feature to a customer for a date engineering never agreed to",
        "two of your engineers each believe they own the same critical component",
        "a partner team's reorg just removed the people you depended on",
        "an underperforming engineer is well-liked and nobody has told them the truth",
    ])
    stake = rng.choice(["a safety-critical launch", "the quarter's headline OKR",
                        "a customer escalation with exec visibility", "your team's on-call health"])
    return {
        "scenario": (f"You are the EM of a perception team. In the middle of {stake}, {tension}. "
                     f"This exercise targets '{comp_name}'. Describe concretely how you handle it: the "
                     f"decision you make, how you communicate it, and how you measure whether it worked. "
                     f"Ground your answer in a real experience if you have one."),
        "expected_reasoning": [
            "Names the actual decision and owns it personally (not 'we would')",
            "Considers and rejects at least one alternative",
            "Addresses the people dimension explicitly (conversations, expectations)",
            "Defines a measurable outcome and a follow-through check",
            "Manages risk to the in-flight commitment",
        ],
        "rubric": [
            RubricItem(criterion="Personal ownership: states the decision THEY made",
                       keywords=["i decided", "i chose", "my decision", "i told", "i set", "i asked",
                                 "i pushed", "i made the call", "i took"]),
            RubricItem(criterion="Considers alternatives and why they were rejected",
                       keywords=["alternative", "instead of", "rejected", "considered", "option",
                                 "could have", "chose over"]),
            RubricItem(criterion="Handles the people/communication dimension explicitly",
                       keywords=["1:1", "one-on-one", "conversation", "feedback", "expectations",
                                 "communicat", "listen", "align", "stakeholder"]),
            RubricItem(criterion="Defines a measurable outcome and follow-through",
                       keywords=["measure", "outcome", "result", "metric", "follow up", "follow-up",
                                 "checked back", "retro"]),
            RubricItem(criterion=f"Engages the specific competency ({', '.join(topics[:3])})",
                       keywords=[t.lower() for t in topics] or ["leadership"]),
            QUANT_RUBRIC, TRADEOFF_RUBRIC,
        ],
        "failure_modes": [
            "Abstract 'best practices' answer with no personal decision",
            "No measurable result — the story ends at the action",
            "Avoiding the uncomfortable conversation entirely",
        ],
        "follow_ups": [
            "What would you have done if your first approach failed?",
            "What did this cost you, and was it worth it?",
            "How did you know your intervention actually caused the improvement?",
        ],
        "linked_tool": None,
    }


def _hill_climb_concept(rng: random.Random, difficulty: int) -> Dict:
    objectives = rng.sample(["detection quality", "safety incident rate", "infra cost",
                             "team velocity", "on-call health", "customer-visible latency"], 3)
    return {
        "scenario": (f"You inherit a system where {objectives[0]}, {objectives[1]}, and {objectives[2]} "
                     f"are all trending the wrong way, and improving any one of them naively damages another. "
                     f"Describe your hill-climbing approach: how you pick the next intervention, how you "
                     f"measure it, when you keep vs revert, and how you avoid local optima and hard-constraint "
                     f"violations (e.g. safety floors)."),
        "expected_reasoning": [
            "Defines a balanced multi-objective view with hard constraints, not a single collapsed score",
            "One intervention at a time with a hypothesis and a measurement window",
            "Keep/revert discipline based on measured effect, not sunk cost",
            "Watches second-order and delayed consequences",
            "Escapes local optima with occasional larger structural bets",
        ],
        "rubric": [
            RubricItem(criterion="Frames it as multi-objective with hard floors/constraints",
                       keywords=["multi-objective", "hard constraint", "floor", "threshold", "guardrail",
                                 "never below", "competing objectives"]),
            RubricItem(criterion="Hypothesis-driven single-change iterations",
                       keywords=["hypothesis", "one change", "single intervention", "iterate", "experiment"]),
            RubricItem(criterion="Keep/revert discipline from measurement",
                       keywords=["revert", "keep", "roll back", "rollback", "measured effect", "kill"]),
            RubricItem(criterion="Considers delayed and second-order effects",
                       keywords=["second-order", "delayed", "lag", "downstream effect", "unintended"]),
            RubricItem(criterion="Has a strategy for local optima",
                       keywords=["local optim", "plateau", "bigger bet", "structural", "step change",
                                 "restart"]),
            QUANT_RUBRIC, TRADEOFF_RUBRIC,
        ],
        "failure_modes": [
            "Optimizing a single collapsed score and violating a safety floor",
            "Changing many things at once — no attribution",
            "Never reverting anything",
        ],
        "follow_ups": [
            "Your last three interventions all measured flat. What now?",
            "An exec demands you optimize velocity only for one quarter. How do you respond?",
        ],
        "linked_tool": None,
    }


def _generic_technical(rng: random.Random, difficulty: int, comp_name: str, topics: List[str]) -> Dict:
    scale = rng.choice(["a fleet of 500 test vehicles", "a petabyte-scale sensor lake",
                        "a 40-engineer perception org", "a nightly retraining pipeline"])
    angle = rng.choice(["design it from scratch", "critique and fix the current design",
                        "explain it to a new senior hire, including the sharp edges",
                        "defend it in an architecture review under hostile questioning"])
    topic_str = ", ".join(topics[:4]) if topics else comp_name
    return {
        "scenario": (f"In the context of {scale}, {angle}: '{comp_name}' ({topic_str}). "
                     f"Be specific about mechanisms, numbers, failure modes, and tradeoffs — "
                     f"not just terminology."),
        "expected_reasoning": [
            f"Correct working definitions of {topic_str}",
            "Mechanisms explained causally, not just named",
            "Concrete failure modes and how to detect them",
            "Quantified reasoning where applicable",
            "Explicit tradeoffs",
        ],
        "rubric": [
            RubricItem(criterion=f"Covers the core concepts ({topic_str})",
                       keywords=[t.lower() for t in topics] or [comp_name.lower()], weight=2.0),
            RubricItem(criterion="Explains mechanisms causally (because/therefore/so that)",
                       keywords=["because", "therefore", "which means", "so that", "leads to",
                                 "causes"]),
            RubricItem(criterion="Names concrete failure modes and their detection",
                       keywords=["fail", "failure mode", "breaks", "degrade", "detect", "edge case"]),
            QUANT_RUBRIC, TRADEOFF_RUBRIC,
        ],
        "failure_modes": [
            "Terminology name-dropping without causal mechanism",
            "No failure modes considered",
            "No numbers anywhere",
        ],
        "follow_ups": [
            f"Which part of {comp_name} do practitioners most often get wrong, and why?",
            "Give a concrete number or threshold you'd use, and defend it.",
        ],
        "linked_tool": None,
    }


FAMILY_BUILDERS = {
    "offline_shadow": _offline_shadow,
    "cusum_drift": _cusum_drift,
    "parallel_inference": _parallel_inference,
    "hill_climb": _hill_climb_concept,
}

# Competency → template family. Anything unlisted uses the generic builder
# (leadership competencies use the leadership scenario builder).
COMPETENCY_FAMILY = {
    "p1.regression_detection": "offline_shadow",
    "p1.failure_analysis": "offline_shadow",
    "p1.cusum": "cusum_drift",
    "p1.model_monitoring": "cusum_drift",
    "p2.parallel_inference": "parallel_inference",
    "p4.hill_climbing": "hill_climb",
}


def generate_exercise(competency_id: str, difficulty: int = 2, seed: Optional[int] = None,
                      store: Optional[Store] = None, use_llm: bool = True) -> Exercise:
    store = store or get_store()
    bp = load_blueprint(store)
    idx = competency_index(bp)
    if competency_id not in idx:
        raise ValueError(f"unknown competency '{competency_id}'")
    comp = idx[competency_id]
    if seed is None:
        seed = random.SystemRandom().randrange(1, 10 ** 9)
    rng = random.Random(seed)

    family = COMPETENCY_FAMILY.get(competency_id)
    if family:
        built = FAMILY_BUILDERS[family](rng, difficulty)
    elif comp.phase == 3:
        family = "leadership_scenario"
        built = _leadership_scenario(rng, difficulty, comp.name, comp.topics)
    else:
        family = "generic_technical"
        built = _generic_technical(rng, difficulty, comp.name, comp.topics)

    scenario = built["scenario"]
    # Optional LLM enrichment of the scenario prose only — the rubric and
    # expected reasoning stay deterministic (they drive scoring).
    if use_llm and llm.llm_enabled():
        enriched = llm.generate_text(
            "Rewrite this interview exercise scenario to be vivid and specific in <=120 words. "
            "Do not change any numbers, do not add new requirements, keep second person:\n\n" + scenario,
            timeout=10.0,
        )
        if enriched and len(enriched) > 40:
            scenario = enriched.strip()

    ex = Exercise(
        competency_id=competency_id,
        difficulty=difficulty,
        prerequisites=comp.prerequisites,
        scenario=scenario,
        expected_reasoning=built["expected_reasoning"],
        evaluation_rubric=built["rubric"],
        common_failure_modes=built["failure_modes"],
        follow_up_questions=built["follow_ups"],
        template_id=f"{family}#{seed % 1000}",
        seed=seed,
        family=family,
        linked_tool=built.get("linked_tool"),
    )
    store.put("exercises", ex.exercise_id, ex)
    return ex


def get_exercise(exercise_id: str, store: Optional[Store] = None) -> Optional[Exercise]:
    raw = (store or get_store()).get("exercises", exercise_id)
    return Exercise(**raw) if raw else None
