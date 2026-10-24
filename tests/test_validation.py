import numpy as np
import pandas as pd

from isagiph.validation import aggregate, validate


def test_validation_reports_spatially_aware_uncertainty_and_keeps_population_weighting():
    df = pd.DataFrame({
        "psgc": [f"{i:010d}" for i in range(8)],
        "province": [f"P{i}" for i in range(8)],
        "region": ["R1"] * 4 + ["R2"] * 4,
        "population": [10, 20, 30, 40, 50, 60, 70, 80],
        "gap_index": np.arange(8, dtype=float),
        "outcome": np.arange(8, 0, -1, dtype=float),
    })
    agg = aggregate(df, "province", ["gap_index", "outcome"])
    assert len(agg) == 8
    result = validate(df, {"project": {"seed": 13},
                           "validation": {"outcomes": ["outcome"], "expected_sign": {"outcome": -1}}})
    assert set(result["level"]) == {"lgu", "province"}
    assert {"spearman_ci95_low", "spearman_ci95_high", "p_value_kind", "ci_method"} <= set(result.columns)
    assert result["consistent"].all()
    assert result.loc[result["level"].eq("lgu"), "ci_method"].iloc[0] == "region-block bootstrap"
