"""STAR Story Box: diagnosis, claim-vs-evidence, competency mapping, evidence storage."""

from sensorflow.hillclimb.star import diagnose_story

GOOD_STORY = (
    "When I joined, the team of 12 engineers owned a perception system whose release process was "
    "failing under a hard quarterly deadline. I was asked to fix the release pipeline. "
    "I decided to build a regression gate, and instead of pausing all launches I chose a staged "
    "rollout; the tech lead disagreed, and I convinced him with data from the last three incidents. "
    "As a result, deploy failures dropped from 14% to 3% within two quarters, and since then we "
    "have sustained the improvement with weekly monitoring."
)

VAGUE_STORY = (
    "My team had some problems with quality. We worked together on improving things and "
    "significantly improved performance. Everyone was happier and things got much better overall."
)


def test_good_story_diagnosed_into_components():
    d = diagnose_story(GOOD_STORY, save_evidence=False)
    by_label = {c.component: c for c in d.components}
    assert by_label["S"].present and by_label["A"].present and by_label["R"].present
    assert d.overall_score >= 4
    passed = {c.check for c in d.checks if c.passed}
    assert {"personal_ownership", "measurable_outcome", "influence_disagreement",
            "constraints", "follow_through"} <= passed


def test_claim_vs_evidence_detector():
    vague = diagnose_story(VAGUE_STORY, save_evidence=False)
    unquantified = [f for f in vague.claim_flags if f.kind == "unquantified_claim"]
    assert unquantified, "must flag 'significantly improved performance' style claims"
    assert any("metric" in f.detail or "before" in f.detail for f in unquantified)

    good = diagnose_story(GOOD_STORY, save_evidence=False)
    measurable = [f for f in good.claim_flags if f.kind == "measurable_evidence"]
    assert measurable, "quantified before/after claims must be recognized as evidence"


def test_unquantified_claims_cap_score_and_produce_strengthen_prompts():
    d = diagnose_story(VAGUE_STORY, save_evidence=False)
    assert d.overall_score <= 3
    assert any("Strengthen evidence" in c for c in d.coaching)
    # weak ownership called out specifically
    ownership = next(c for c in d.checks if c.check == "personal_ownership")
    assert not ownership.passed


def test_story_maps_to_phase3_competencies():
    d = diagnose_story(GOOD_STORY, save_evidence=False)
    ids = {c["competency_id"] for c in d.competencies}
    assert "p3.conflict_resolution" in ids  # "disagreed"
    assert all(cid.startswith("p3.") for cid in ids)
    assert all(c["reason"] for c in d.competencies)


def test_saved_as_evidence_artifact(isolated_store):
    d = diagnose_story(GOOD_STORY, user_id="u1", save_evidence=True, store=isolated_store)
    assert d.evidence_id
    raw = isolated_store.get("evidence", d.evidence_id)
    assert raw["artifact_type"] == "star_story"
    assert raw["competency_ids"]
    assert raw["score"] == d.overall_score
