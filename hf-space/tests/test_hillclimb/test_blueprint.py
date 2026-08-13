"""Blueprint & competency-graph integrity."""

from sensorflow.hillclimb.blueprint import (competency_index, downstream_map,
                                            display_dimensions_for, load_blueprint,
                                            seed_blueprint, validate_graph)
from sensorflow.hillclimb.models import Dimension, get_store


def test_four_phases_with_full_structure():
    bp = seed_blueprint()
    assert [p.phase for p in bp.phases] == [1, 2, 3, 4]
    for p in bp.phases:
        assert p.objective
        assert p.topics and p.skills and p.exercises and p.assessments
        assert p.completion_criteria


def test_marked_reconstructed_from_spec_and_seeded_editable():
    store = get_store()
    bp = load_blueprint(store)
    assert bp.source == "reconstructed-from-spec"
    # persisted as an editable data structure, not hardcoded prose
    raw = store.get("blueprint", "active")
    assert raw is not None and raw["source"] == "reconstructed-from-spec"
    # editing the stored copy is what subsequent loads see
    raw["version"] = 2
    store.put("blueprint", "active", raw)
    assert load_blueprint(store).version == 2


def test_no_prerequisite_cycles_and_no_dangling_ids():
    bp = seed_blueprint()
    assert validate_graph(bp) == []


def test_every_competency_tagged_with_exactly_one_dimension():
    bp = seed_blueprint()
    for c in bp.competencies:
        assert isinstance(c.dimension, Dimension)
    # all four dimensions are represented and tracked separately
    dims = {c.dimension for c in bp.competencies}
    assert dims == set(Dimension)


def test_key_prerequisite_edge_distributed_fundamentals_to_parallel_inference():
    idx = competency_index(seed_blueprint())
    assert "p2.distributed_fundamentals" in idx["p2.parallel_inference"].prerequisites


def test_downstream_map_transitive():
    bp = seed_blueprint()
    down = downstream_map(bp)
    # distributed fundamentals transitively unlocks the phase-4 capstone
    assert "p4.hill_climbing" in down["p2.distributed_fundamentals"]
    # capstone unlocks nothing
    assert down["p4.hill_climbing"] == set()


def test_display_dimensions_cover_all_six():
    bp = seed_blueprint()
    covered = set()
    for c in bp.competencies:
        covered |= set(display_dimensions_for(c))
    assert covered == {"Technical Depth", "System Design", "Execution",
                       "Leadership", "Communication", "Safety/Risk"}
