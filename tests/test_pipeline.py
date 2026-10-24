import json

import pandas as pd
import pytest
import yaml

from isagiph.schema import validate


def test_schema_rejects_points_outside_philippines():
    df = pd.DataFrame({"point_id": ["x"], "psgc": ["1"], "lat": [40.0], "lon": [121.0], "population": [5]})
    with pytest.raises(ValueError):
        validate(df, "points")


def test_demo_pipeline_smoke(tmp_path):
    from isagiph import pipeline
    # Keep this end-to-end test hermetic: never write generated demo data or
    # overwrite the repository's real findings report during pytest.
    cfg = yaml.safe_load(pipeline.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"] = {name: str(tmp_path / name) for name in cfg["paths"]}
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    pipeline.main(["--demo", "--runs", "5", "--config", str(config_path)])
    processed = pipeline.load_config(config_path)["paths"]["processed"]
    res = pd.read_csv(processed / "lgu_results.csv")
    assert {"HVS", "HCS", "gap_index", "lisa_cluster", "priority_freq"} <= set(res.columns)
    assert res["gap_index"].between(-100, 100).all()
    # Additive diagnostics are written without removing the existing pipeline outputs.
    for name in ("ml_predictions.csv", "ml_importance.csv", "ml_metrics.json", "ml_region_metrics.csv",
                 "ml_subgroup_metrics.csv", "ml_validation.md", "run_meta.json"):
        assert (processed / name).exists()
    met = json.loads((processed / "ml_metrics.json").read_text())
    assert "mean_baseline" in met["models"] and "nested_selection" in met
    assert met["interval_formal_conformal_guarantee"] is False
    run_meta = json.loads((processed / "run_meta.json").read_text())
    assert "provenance" in run_meta and "sha256_files" in run_meta["provenance"]
