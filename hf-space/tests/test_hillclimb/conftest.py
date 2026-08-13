import pytest

from sensorflow.hillclimb.models import reset_store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Every test runs against a fresh store and with the LLM disabled, so the
    deterministic rule-based paths are what is under test."""
    monkeypatch.setenv("HILLCLIMB_DISABLE_LLM", "1")
    store = reset_store(tmp_path / "hillclimb")
    yield store
    reset_store(tmp_path / "hillclimb-teardown")


# ---------------------------------------------------------- shared answer text

# Covers nearly every rubric criterion of the offline/shadow exercise family,
# with quantified claims and explicit tradeoffs.
STRONG_TECH_ANSWER = (
    "First I would check for a metric definition mismatch: offline eval and shadow monitoring may "
    "use a different IoU threshold or matching threshold, so the two numbers are not the same metric. "
    "Second, distribution shift: live traffic differs from the eval set, so the population changed. "
    "Third, sampling bias in how the eval set was built — it may not be representative. "
    "Fourth, training-serving skew: the feature pipeline can produce a feature mismatch at serving time. "
    "Fifth, a serving mismatch: preprocessing or quantization differences in the inference stack. "
    "I would also test statistical significance — the shadow delta may be noise, so I would compute a "
    "95% confidence interval given the sample size. A true regression remains possible, so I would "
    "slice metrics by scenario, run a matched A/B experiment, and re-score identical frames in both stacks. "
    "The tradeoff: this delays launch by 3 days, at the cost of schedule, but avoids shipping a "
    "genuinely worse model."
)

# Long, polished-sounding, but covers no rubric concept and has no numbers.
VERBOSE_WAFFLE = (
    "This is truly an interesting and important challenge, and in situations like this one it is "
    "essential to take a holistic, thoughtful, and collaborative approach that considers the "
    "perspectives of all involved parties. I believe the most valuable thing a leader can do is to "
    "bring people together, encourage open communication, and ensure that everyone feels heard and "
    "aligned around a shared vision of success. It is also important to remember that great teams "
    "are built on trust, and trust is built through consistency, empathy, and transparency. "
    "I would begin by gathering the relevant people in a room and facilitating a constructive "
    "discussion about the situation, its causes, and the potential paths forward, always keeping an "
    "open mind and remaining flexible as new information emerges. Ultimately, what matters most is "
    "that we move forward together with confidence, clarity, and a renewed sense of purpose, and "
    "that we continue to learn and grow as an organization through every challenge we encounter. "
    "By staying positive, communicating openly, and supporting one another, I am confident that "
    "we will arrive at a good outcome for the business, for the team, and for our users, and that "
    "this challenge will make us stronger as a group going forward."
)

CONCISE_PRECISE_ANSWER = (
    "Likely causes: a metric definition mismatch between the two pipelines, distribution shift "
    "between eval data and live traffic, or sampling bias in the eval set. Before reacting I would "
    "check statistical significance of the shadow delta."
)

STRONG_LEADERSHIP_ANSWER = (
    "When our launch slipped I decided to pause the feature work. Instead of quietly absorbing the "
    "risk, I considered delaying the launch or cutting scope, and rejected the delay because it "
    "punished the wrong team. I held a 1:1 with the tech lead, gave direct feedback, and aligned "
    "the stakeholders on the new plan. We agreed to measure the result weekly, and the outcome was "
    "a 30% drop in escaped regressions within 2 months. The tradeoff: we gave up one roadmap item "
    "at the cost of short-term velocity."
)

WEAK_ANSWER = "Honestly not sure. Probably fine either way."
