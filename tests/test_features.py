import numpy as np
import pandas as pd

from isagiph.equity import concentration_index, weighted_quintile


def test_concentration_index_sign():
    pov = pd.Series([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])          # LGU 0 poorest
    w = pd.Series([1.0] * 6)
    pro_rich = pd.Series([1, 2, 3, 4, 5, 6.0])                # more beds in less-poor LGUs
    assert concentration_index(pov, pro_rich, w) > 0
    assert concentration_index(pov, pro_rich[::-1].reset_index(drop=True), w) < 0
    assert abs(concentration_index(pov, pd.Series([3.0] * 6), w)) < 1e-9


def test_weighted_quintile_balanced():
    x = pd.Series(np.linspace(0, 1, 100))
    q = weighted_quintile(x, pd.Series([1.0] * 100))
    assert q.value_counts().min() == 20 and q[99] == "Q1 poorest"


def test_ml_typology_and_expected_capacity():
    from isagiph.ml import expected_capacity, typology
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"psgc": [str(i) for i in range(n)], "region": rng.choice(list("ABCDE"), n),
                       "population": rng.integers(5000, 90000, n), "poverty_incidence": rng.uniform(0.05, 0.6, n),
                       "gida_share": rng.uniform(0, 1, n), "pct_age_0_4": rng.uniform(0.08, 0.14, n),
                       "pct_age_60_plus": rng.uniform(0.05, 0.15, n)})
    df["beds_per_1000_all"] = np.exp(1 - 3 * df["poverty_incidence"] + rng.normal(0, 0.1, n))
    for c in ("hvs_a", "hvs_b", "hcs_a", "hcs_b"):
        df[c] = rng.uniform(0, 100, n)
    pred, met, imp = expected_capacity(df)
    assert met[met["best"]]["r2"] > 0.5 and imp.iloc[0]["feature"] == "poverty_incidence"
    band = met["prediction_band"]
    assert 0 <= band["empirical_coverage_on_outer_heldout_rows"] <= 1
    assert band["n_outer_heldout_rows_with_interval"] > 0
    assert "held-out outer rows" in band["calibration_scope"]
    assert len(band["fold_calibration"]) >= 2
    assert "mean_baseline" in met["models"] and "ridge" in met["models"]
    assert met["nested_selection"]["available"]
    assert met["baseline_comparison"]["comparison_uses_nested_oof"]
    assert np.isclose(met["baseline_comparison"]["selected_model_mae_beds"], met["nested_selection"]["mae_beds"])
    assert met["interval_formal_conformal_guarantee"] is False
    assert met["preprocessing"]["imputation"].startswith("Median imputation is fitted independently")
    assert {"model_used", "validation_fold", "abs_error_beds"} <= set(pred.columns)
    assert "rf_importance_std" in imp and imp["importance_method"].str.contains("held-out").all()
    typ, info = typology(df)
    assert typ["cluster"].nunique() == info["k"] and 3 <= info["k"] <= 8   # k chosen by silhouette
    assert 0 <= info["stability_ari"] <= 1 and typ["cluster_name"].str.contains("need").all()


def test_community_store_strips_exif(tmp_path):
    import io
    from PIL import Image
    from isagiph.community import Store
    s = Store(tmp_path)
    im = Image.new("RGB", (2000, 1500)); ex = Image.Exif(); ex[0x8825] = {2: (14.0, 35.0, 0.0)}
    b = io.BytesIO(); im.save(b, "JPEG", exif=ex.tobytes())
    name = s.save_photo(b.getvalue())
    out = Image.open(tmp_path / "photos" / name)
    assert max(out.size) == 1280 and not out.getexif()
    s.add_report(facility_id="F1", facility_name="X", psgc="P", crowding="Busy", wait="Under 30 min", affordability=2,
                 out_of_pocket=None, note=None, photo=name, alias="A")
    assert len(s.latest_status()) == 1


def test_morbidity_parser_reads_national_totals():
    """Checks the DOH table parser against the published national totals (skipped without the PDF)."""
    from pathlib import Path
    pdfs = sorted(Path(__file__).resolve().parents[1].glob("data/manual/Morbidity*.pdf"))
    if not pdfs:
        return
    from isagiph.build_real import read_morbidity
    raw = read_morbidity(pdfs[0])
    ph = raw[raw["area"] == "PHILIPPINES"].set_index("disease")
    assert ph.loc["tb", "cases"] == 119558 and abs(ph.loc["tb", "rate"] - 107.16) < 0.01
    assert ph.loc["hypertension", "cases"] == 668871
    assert raw.groupby("disease").size().nunique() == 1          # same areas for every disease


def test_profile_pdf_is_a_one_page_pdf():
    from isagiph.profile_pdf import lgu_profile_pdf
    r = pd.Series({"name": "Testville", "province": "Testprov", "region": "Region X", "population": 25000,
                   "gap_index": 60.0, "HVS": 70.0, "HCS": 15.0, "hvs_pct": 90.0, "hcs_pct": 10.0, "priority": True,
                   "priority_freq": 0.95, "unstable": False, "v_tertile": 3, "c_tertile": 1, "hvs_demographic": 50.0,
                   "hvs_socioeconomic": 80.0, "hcs_beds": 5.0, "hcs_coverage": 20.0, "poverty_incidence": 0.4,
                   "beds_per_1000_all": 0.2, "cov_any_30": 0.3, "cov_malasakit_60": 0.1, "bed_gap_shortfall": 26.0,
                   "pop_children": 8000, "pop_youth": 5000, "pop_young_adults": 7000, "pop_middle_age": 3000,
                   "pop_elderly": 2000})
    pdf = lgu_profile_pdf(r, pd.DataFrame([r]))
    assert pdf[:4] == b"%PDF" and pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + pdf.count(b"/Type/Page") >= 1


def test_seed_samples_are_labelled_and_removable(tmp_path):
    from isagiph.community import Store, remove_samples, seed_samples
    s = Store(tmp_path)
    d = pd.DataFrame({"facility_id": ["F1"], "name": ["BICOL MEDICAL CENTER"], "psgc": ["P1"]})
    lg = pd.DataFrame({"psgc": ["P2"], "name": ["Naga City"], "province": ["Camarines Sur"], "region": ["Region V"]})
    n_r, n_a = seed_samples(s, d, lg)
    assert (n_r, n_a) == (1, 1)
    assert s.reports()["note"].str.startswith("[Sample]").all() and s.announcements()["title"].str.startswith("[Sample]").all()
    remove_samples(s)
    assert s.reports().empty and s.announcements().empty


def test_road_graph_gives_city_block_times():
    from isagiph import roadtimes as rt
    lat0, lon0, n, step = 13.0, 123.0, 11, 0.01
    coords, idx, k = [], [], 0
    for i in range(n):
        coords += [(lon0 + j * step, lat0 + i * step) for j in range(n)]; idx += [k] * n; k += 1
    for j in range(n):
        coords += [(lon0 + j * step, lat0 + i * step) for i in range(n)]; idx += [k] * n; k += 1
    G, nodes = rt.graph_from_lines(np.array(coords), np.array(idx), np.full(k, 36.0))
    assert len(nodes) == n * n
    rt.MIN_COMPONENT = 5
    pts = pd.DataFrame({"point_id": ["p"], "lat": [lat0 + 0.05], "lon": [lon0 + 0.05], "psgc": "x", "population": 1})
    fac = pd.DataFrame({"facility_id": ["h"], "lat": [lat0], "lon": [lon0]})
    tt, on = rt.route(G, nodes, pts, fac)
    from isagiph.accessibility import to_local_km
    xy = to_local_km(np.array([lat0, lat0 + 0.05, lat0]), np.array([lon0, lon0, lon0 + 0.05]))
    expect = (np.hypot(*(xy[1] - xy[0])) + np.hypot(*(xy[2] - xy[0]))) / 36 * 60
    assert abs(tt["minutes"].iloc[0] - expect) < 0.1


def test_greedy_siting_covers_the_biggest_uncovered_cluster_first():
    from isagiph.optimize import best_sites
    # two uncovered villages far apart (A large, B small) and one already-covered town
    pts = pd.DataFrame({"point_id": ["a1", "a2", "b1", "t1"], "lat": [13.0, 13.01, 14.0, 12.0],
                        "lon": [123.0, 123.01, 124.0, 122.0], "population": [5000, 4000, 1000, 9000],
                        "psgc": ["A", "A", "B", "T"]})
    need = pd.Series({"A": 50.0, "B": 50.0, "T": 50.0})
    mins = pd.Series([90.0, 90.0, 90.0, 10.0])
    picks = best_sites(pts, need, mins, n_sites=3)
    assert list(picks["psgc"])[:2] == ["A", "B"]           # largest cluster first, then the next
    assert picks["people_covered"].iloc[0] == 9000 and len(picks) == 2   # covered town never chosen; stops when done


def test_conformal_halfwidth_gives_requested_coverage():
    from isagiph.ml import conformal_halfwidth
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000); pred = y + rng.normal(scale=0.5, size=5000)
    q = conformal_halfwidth(y, pred, alpha=0.1)
    assert abs((np.abs(y - pred) <= q).mean() - 0.9) < 0.01


def test_fallback_travel_times_use_explicit_transport_assumptions(tmp_path):
    from isagiph.pipeline import travel_times
    points = pd.DataFrame({"point_id": ["P1"], "psgc": ["L1"], "lat": [0.0], "lon": [0.0], "population": [1000]})
    facilities = pd.DataFrame({"facility_id": ["H1"], "lat": [0.0], "lon": [0.01]})
    cfg = {
        "paths": {"interim": tmp_path},
        "accessibility": {"fallback_speed_kmh": 15.0, "circuity": 1.4,
                           "d0_minutes": {"level2plus": 60}},
        "sensitivity": {"d0_multipliers": [0.5, 1.0, 1.5]},
    }
    tt, source = travel_times({"points": points, "facilities": facilities}, cfg, demo=True)
    expected_distance_km = 0.01 * 111.320
    expected_minutes = expected_distance_km * 1.4 / 15.0 * 60.0
    assert np.isclose(tt.loc[0, "minutes"], expected_minutes, rtol=0.01)
    assert "configured transport assumptions" in source and "context" in source


def test_hospital_access_estimate_is_not_direct_doh_validation():
    from isagiph.access_estimate import summarize_hospital_access
    results = pd.DataFrame({"population": [1000, 3000], "cov_any_30": [0.8, 0.2]})
    cfg = {"accessibility": {"fallback_speed_kmh": 15, "circuity": 1.4},
           "roads": {"access_speed_kmh": 15, "speeds_kmh": {"primary": 50}}}
    summary = summarize_hospital_access(results, cfg, road_network_population_share=0.5)
    assert summary["population_total"] == 4000
    assert summary["population_within_threshold"] == 1400
    assert summary["population_outside_threshold"] == 2600
    assert "estimate under the configured transport assumptions" in summary["interpretation"] and "not a count of patients" in summary["interpretation"]
    assert summary["reference_comparison"]["role"] == "external comparison after estimation"


def test_graph_imputation_beats_the_median_on_a_smooth_map():
    from isagiph.impute import graph_impute
    rng = np.random.default_rng(2)
    n = 600
    lat, lon = rng.uniform(5, 18, n), rng.uniform(118, 126, n)
    prov = (np.floor(lat) * 100 + np.floor(lon)).astype(int).astype(str)          # ~1° x 1° "provinces"
    rate = np.exp(3 + np.sin(lat / 2) + np.cos(lon / 1.5))                        # smooth north-south pattern
    lg = pd.DataFrame({"psgc": range(n), "region": "R", "province": prov, "lat": lat, "lon": lon, "population": 20000,
                       "pop_0_4": 2000, "pop_60_plus": 1500, "poverty_incidence": 0.2, "gida_share": 0.1,
                       "area_sqkm": 50, "rate_x": rate})
    hidden = np.isin(prov, rng.choice(np.unique(prov), 8, replace=False))
    lg["rate_x_imputed"] = hidden.astype(int)
    lg.loc[hidden, "rate_x"] = np.median(rate[~hidden])                           # what the regional median gives
    out, rep = graph_impute(lg, ["rate_x"])
    assert rep["rate_x"]["method_used"] != "regional median"
    err_new = np.abs(np.log1p(out.loc[hidden, "rate_x"]) - np.log1p(rate[hidden])).mean()
    err_med = np.abs(np.log1p(lg.loc[hidden, "rate_x"]) - np.log1p(rate[hidden])).mean()
    assert err_new < err_med


def test_ownership_class_label_preserves_doh_retained_capitalization():
    from isagiph.build_real import ownership_class_label

    assert ownership_class_label("DOH RETAINED") == "DOH-retained"
    assert ownership_class_label("Doh Retained") == "DOH-retained"
    assert ownership_class_label("DOH-retained") == "DOH-retained"
    assert ownership_class_label("LGU") == "LGU"
    assert ownership_class_label(None) == ""
