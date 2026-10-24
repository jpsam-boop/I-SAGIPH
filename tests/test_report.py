"""Report logic: bottleneck classification, equity confidence interval, and the policy brief."""
import numpy as np
import pandas as pd

from isagiph.report import bottlenecks, concentration_ci, write_policy_brief


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pov = rng.uniform(0.05, 0.6, n)
    return pd.DataFrame({
        "psgc": [f"P{i}" for i in range(n)], "name": [f"Town {i}" for i in range(n)], "province": "Prov",
        "region": np.where(np.arange(n) % 2, "Region A", "Region B"), "population": rng.integers(5000, 90000, n),
        "poverty_incidence": pov,
        # beds fall as poverty rises -> beds concentrated in less-poor areas (positive index)
        "beds_per_1000_all": np.clip(2.0 - 2.5 * pov + rng.normal(0, 0.1, n), 0, None),
        "accessible_beds": 100.0, "cov_any_30": rng.uniform(0, 1, n), "cov_malasakit_60": rng.uniform(0, 1, n),
        "priority": rng.uniform(0, 1, n) < 0.3, "unstable": False, "bed_gap_shortfall": rng.uniform(0, 50, n)})


def test_bottleneck_is_distance_when_most_residents_are_far():
    df = _toy()
    b = bottlenecks(df, 1.25)
    assert set(b["bottleneck"]) <= {"Distance", "Bed supply"}
    assert (b.loc[b["cov_any_30"] < 0.5, "bottleneck"] == "Distance").all()
    assert (b.loc[b["cov_any_30"] >= 0.5, "bottleneck"] == "Bed supply").all()


def test_concentration_ci_has_the_right_sign_and_contains_the_estimate():
    est, lo, hi = concentration_ci(_toy(), n_boot=200)
    assert est > 0 and lo <= est <= hi            # beds favour less-poor areas in the toy data
    assert lo > 0                                 # and the interval excludes 0


def test_policy_brief_is_written(tmp_path):
    out = tmp_path / "policy_brief.md"
    write_policy_brief(_toy(), {"mode": "real", "beds_per_1000": 1.25, "instability_threshold": 0.2}, out)
    text = out.read_text(encoding="utf-8")
    assert "where to act first" in text and "Top three per region" in text
