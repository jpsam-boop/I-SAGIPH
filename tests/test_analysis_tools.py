import numpy as np
import pandas as pd

from isagiph.analysis_tools import disruption_summary, regression_metrics, relationship_statistics


def test_regression_metrics_are_in_target_units():
    out = regression_metrics([1, 2, 3], [2, 2, 5])
    assert out["n"] == 3
    assert np.isclose(out["mae"], 1.0)
    assert np.isclose(out["rmse"], np.sqrt(5 / 3))
    assert np.isclose(out["bias"], 1.0)


def test_relationship_statistics_reports_region_bootstrap():
    df = pd.DataFrame({
        "x": np.arange(12, dtype=float), "y": np.arange(12, dtype=float) * 2,
        "region": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
    })
    out = relationship_statistics(df, "x", "y", n_boot=50, seed=1)
    assert out["n"] == 12
    assert out["rho"] > 0.99
    assert out["bootstrap_method"] == "region-block bootstrap"
    assert out["rho_ci_low"] > 0.8


def test_disruption_summary_counts_population_crossing_access_threshold():
    points = pd.DataFrame({
        "point_id": ["p1", "p2", "p3"], "psgc": ["A", "A", "B"],
        "population": [100, 200, 300],
    })
    before = pd.DataFrame({"point_id": ["p1", "p2", "p3"], "facility_id": ["f1"] * 3,
                           "minutes": [10.0, 25.0, 20.0]})
    after = pd.DataFrame({"point_id": ["p1", "p2"], "facility_id": ["f2", "f2"],
                          "minutes": [20.0, 40.0]})
    by_lgu, total = disruption_summary(points, before, after, threshold_minutes=30)
    assert total["population_losing_reach"] == 500
    assert total["population_within_before"] == 600
    assert total["population_within_after"] == 100
    a = by_lgu.set_index("psgc")
    assert a.loc["A", "population_losing_reach"] == 200
    assert a.loc["B", "population_losing_reach"] == 300
    assert a.loc["B", "reachable_share_after"] == 0
    assert total["route_coverage_share_before"] == 1.0
    assert np.isclose(total["route_coverage_share_after"], 0.5)
    assert total["population_without_route_record_after"] == 300


def test_three_analytical_workspaces_replace_old_data_models_navigation():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    app = (root / "app" / "app.py").read_text(encoding="utf-8")
    analyze = app.split('"Analyze": [', 1)[1].split('],', 1)[0]
    assert analyze.count('page(') == 3
    for filename, title in (
        ("regression_explorer.py", "Regression Explorer"),
        ("disruption.py", "Crisis Impact"),
        ("relationship_analyzer.py", "Health Relationships"),
    ):
        assert (root / "app" / "views" / filename).exists()
        assert title in analyze
    assert 'page("equity.py"' not in analyze
    assert 'page("insights.py"' not in analyze
    assert 'page("disruption.py"' not in app.split('"Explore": [', 1)[1].split('],', 1)[0]


def test_equity_content_is_embedded_in_health_relationships_page():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    rel = (root / "app" / "views" / "relationship_analyzer.py").read_text(encoding="utf-8")
    panel = (root / "app" / "views" / "equity_panel.py").read_text(encoding="utf-8")
    assert 'st.tabs(["Relationships", "Equity & Fairness"])' in rel
    assert "render_equity_panel(res)" in rel
    assert "concentration_index" in panel and "Geographically isolated communities" in panel


def test_adjusted_relationship_reports_standardized_association_and_region_cluster_interval():
    from isagiph.analysis_tools import adjusted_relationship_statistics

    rng = np.random.default_rng(7)
    rows = []
    for region_id in range(12):
        for _ in range(5):
            x = float(rng.normal())
            poverty = float(rng.normal())
            y = 0.8 * x + 0.4 * poverty + float(rng.normal(scale=0.25))
            rows.append({"x": x, "y": y, "poverty_incidence": poverty, "region": f"R{region_id:02d}"})
    out = adjusted_relationship_statistics(pd.DataFrame(rows), "x", "y", ["poverty_incidence"])
    assert out["available"] is True
    assert out["n"] == 60
    assert out["n_regions"] == 12
    assert out["standardized_beta"] > 0.5
    assert np.isfinite(out["ci95_low"])
    assert np.isfinite(out["ci95_high"])
    assert "region-cluster robust" in out["inference_method"]


def test_adjusted_relationship_suppresses_cluster_interval_with_too_few_regions():
    from isagiph.analysis_tools import adjusted_relationship_statistics

    rng = np.random.default_rng(3)
    x = rng.normal(size=36)
    poverty = rng.normal(size=36)
    df = pd.DataFrame({
        "x": x, "y": 1.5 * x + 0.8 * poverty + rng.normal(scale=0.5, size=36),
        "poverty_incidence": poverty, "region": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
    })
    out = adjusted_relationship_statistics(df, "x", "y", ["poverty_incidence"])
    assert out["available"] is True
    assert np.isfinite(out["standardized_beta"])
    assert np.isnan(out["ci95_low"])
    assert "below minimum" in out["inference_method"]


def test_disruption_summary_exposes_route_record_coverage_separately_from_reach():
    points = pd.DataFrame({
        "point_id": ["p1", "p2", "p3"], "psgc": ["A", "A", "B"],
        "population": [100, 200, 300],
    })
    before = pd.DataFrame({"point_id": ["p1", "p2", "p3"], "facility_id": ["f1"] * 3,
                           "minutes": [10.0, 25.0, 20.0]})
    after = pd.DataFrame({"point_id": ["p1", "p2"], "facility_id": ["f2", "f2"],
                          "minutes": [20.0, 40.0]})
    by_lgu, totals = disruption_summary(points, before, after, threshold_minutes=30)
    assert totals["route_coverage_share_before"] == 1.0
    assert np.isclose(totals["route_coverage_share_after"], 0.5)
    assert totals["population_without_route_record_after"] == 300
    a = by_lgu.set_index("psgc")
    assert np.isclose(a.loc["A", "route_coverage_share_after"], 1.0)
    assert a.loc["B", "route_coverage_share_after"] == 0.0


def test_relationship_statistics_preserves_lgu_metadata_and_population_weighting():
    df = pd.DataFrame({
        "psgc": [str(i) for i in range(12)], "name": [f"LGU {i}" for i in range(12)],
        "province": ["P"] * 12, "population": np.arange(100, 1300, 100),
        "region": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "x": np.arange(12, dtype=float), "y": np.arange(12, dtype=float) * 2 + 1,
    })
    out = relationship_statistics(df, "x", "y", weights_col="population", n_boot=20, seed=3)
    d = out["data"]
    assert {"psgc", "name", "province", "population"}.issubset(d.columns)
    assert d["population"].ndim == 1
    assert out["n"] == 12
