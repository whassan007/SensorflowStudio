"""Statistical agreement tests: each statistic used correctly."""

import math

from sensorflow.evaluation.graders import cohens_kappa, fleiss_kappa, krippendorff_alpha


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_cohens_kappa_known_value():
    # Classic worked example: po=0.7, pe=0.5 -> kappa=0.4
    a = ["yes"] * 25 + ["yes"] * 15 + ["no"] * 15 + ["no"] * 45
    b = ["yes"] * 25 + ["no"] * 15 + ["yes"] * 15 + ["no"] * 45
    k = cohens_kappa(a, b)
    po = 0.7
    pe = 0.4 * 0.4 + 0.6 * 0.6
    expected = (po - pe) / (1 - pe)
    assert math.isclose(k, expected, abs_tol=1e-9)


def test_cohens_kappa_chance_level_is_zero():
    # Independent raters with 50/50 marginals and po == pe -> kappa == 0
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert math.isclose(cohens_kappa(a, b), 0.0, abs_tol=1e-9)


def test_fleiss_kappa_perfect():
    ratings = [{"cat": 4} for _ in range(10)]
    assert fleiss_kappa(ratings) == 1.0


def test_fleiss_kappa_known_direction():
    # High agreement should beat mixed agreement.
    high = [{"a": 4} for _ in range(8)] + [{"b": 4} for _ in range(8)]
    mixed = [{"a": 2, "b": 2} for _ in range(16)]
    assert fleiss_kappa(high) > fleiss_kappa(mixed)
    assert fleiss_kappa(mixed) < 0.1


def test_krippendorff_alpha_perfect_with_missing_data():
    # Three raters, one has missing values; all present values agree.
    data = [
        ["a", "b", "a", "a"],
        ["a", "b", "a", None],
        [None, "b", "a", "a"],
    ]
    assert krippendorff_alpha(data) == 1.0


def test_krippendorff_alpha_disagreement_lowers_alpha():
    agree = [["a", "b", "a"], ["a", "b", "a"]]
    disagree = [["a", "b", "a"], ["b", "a", "b"]]
    assert krippendorff_alpha(agree) > krippendorff_alpha(disagree)
    assert krippendorff_alpha(disagree) < 0.0  # systematic disagreement
