"""Data models: expected capacity (regression with spatial cross-validation) and an LGU typology (clustering)."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from common import show_table, BLUE, NAVY, PROC, RED, footer, page_header, require_data, section

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("Data models", "What the models tell us",
            "Models estimate an E2SFCA-derived accessibility measure and group LGUs into types with similar needs and gaps. "
            "Regional holdouts, train-fold preprocessing, baselines and uncertainty diagnostics help show where results are reliable.")
if not (PROC / "ml_predictions.csv").exists():
    st.info("Model results are not available.")
    st.stop()
pred = pd.read_csv(PROC / "ml_predictions.csv", dtype={"psgc": str})
imp = pd.read_csv(PROC / "ml_importance.csv")
met = json.loads((PROC / "ml_metrics.json").read_text())
df = res.merge(pred, on="psgc")

section("Model 1 · Regression", "Expected hospital capacity",
        "Predicts accessible beds per 1,000 from poverty, isolation, age structure, density and population size. "
        "Regions are kept intact across validation folds, and preprocessing is learned only from each training fold. "
        "The target is a modelled accessibility indicator, not observed patient use or a causal effect.")
models = met.get("models", {k: {**met[k], "name": k.replace("_", " ").title()} for k in ("linear", "random_forest")})
mc = pd.DataFrame([{"Model": v["name"], "R² (log scale; held-out regions)": round(v["r2"], 3),
                    "MAE (beds per 1,000)": round(v["mae_beds"], 2),
                    "Top candidate*": "✓" if k == met["best"] else ""} for k, v in models.items()])
st.markdown("**Model comparison** (same region-grouped cross-validation for every model; the mean baseline is included). *Top candidate is ranked on pooled outer-fold scores and is exploratory; use nested validation as the primary estimate of the selection procedure.*")
show_table(mc, hide_index=True, use_container_width=True)
r2 = met[met["best"]]["r2"]
nested = met.get("nested_selection", {})
baseline = met.get("baseline_comparison", {})
st.info(f"**Primary model-selection estimate:** nested regional validation gives {nested.get('r2_log', float('nan')):.0%} R² "
        f"and {nested.get('mae_beds', float('nan')):.2f} beds per 1,000 MAE. The top single candidate's pooled outer-fold R² is {r2:.0%}; "
        "because this candidate is identified after comparing those scores, its own score may be optimistic. "
        f"The nested-selection MAE is {baseline.get('selected_model_mae_beds', float('nan')):.2f}, compared with "
        f"{baseline.get('baseline_mae_beds', float('nan')):.2f} for the mean baseline "
        f"(improvement {baseline.get('mae_improvement_beds', float('nan')):+.2f} beds per 1,000). "
        "All scores target the derived E2SFCA accessible-bed measure; they do not establish causality or directly measure staffed beds, service quality or patient outcomes.")
st.caption("LGU-level expectations use nested regional cross-fitting: model family selection happens inside the training folds, "
            "then predictions are made for held-out regions. Candidate model scores are not a substitute for prospective validation.")
c1, c2 = st.columns([1, 1.2])
with c1:
    st.markdown("**Which characteristics matter most** (random forest, held-out permutation importance)")
    st.altair_chart(alt.Chart(imp).mark_bar(color=NAVY, cornerRadiusEnd=4).encode(
        x=alt.X("rf_importance:Q", title="Importance (drop in fit when shuffled)"),
        y=alt.Y("label:N", sort="-x", title=None)).properties(height=240), use_container_width=True)
    st.caption("Permutation importance shows how much held-out model performance worsens when one feature is shuffled. Larger values indicate greater predictive reliance, not a causal effect. Prioritise data-quality checks for influential features.")
    st.markdown("**Direction of each effect** (linear model, standardised coefficients)")
    st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("linear_coef_std:Q", title="← less capacity · more capacity →"), y=alt.Y("label:N", sort="-x", title=None),
        color=alt.condition("datum.linear_coef_std > 0", alt.value(BLUE), alt.value(RED))).properties(height=240),
        use_container_width=True)
    st.caption("Standardised coefficients show the direction and relative size of a linear association while other included features are held fixed. Positive values point toward higher model-estimated accessible capacity; negative values point toward lower capacity. Correlated features can share or reverse apparent effects, so these are not causal effects or direct intervention rankings.")
with c2:
    st.markdown("**Actual vs expected capacity** (each dot is an LGU)")
    sc = df.assign(actual=df["actual_beds_per_1000"] + 0.01, expected=df["expected_beds_per_1000"] + 0.01)
    st.altair_chart(alt.Chart(sc).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("expected:Q", scale=alt.Scale(type="log"), title="Expected beds per 1,000 (model)"),
        y=alt.Y("actual:Q", scale=alt.Scale(type="log"), title="Actual beds per 1,000"),
        color=alt.Color("priority:N", title="Priority area", scale=alt.Scale(range=["#9fb3cf", RED])),
        tooltip=[alt.Tooltip("name:N", title="City/municipality"), alt.Tooltip("province:N", title="Province"),
                 alt.Tooltip("actual_beds_per_1000:Q", format=".2f", title="Actual beds per 1,000"),
                 alt.Tooltip("expected_beds_per_1000:Q", format=".2f", title="Expected beds per 1,000")]).properties(height=520).interactive(),
        use_container_width=True)
conf = met.get("conformal")
if conf and "below_expected_90" in df:
    st.markdown(f"**Below the empirical {conf['level']:.0%} spatial-validation band**")
    coverage = conf.get("empirical_coverage_on_outer_heldout_rows", conf.get("empirical_coverage", float("nan")))
    if conf.get("available", True) and pd.notna(coverage):
        st.write(f"Band coverage was {coverage:.0%} on outer-held-out rows with a calculable interval (n={conf.get('n_outer_heldout_rows_with_interval', 0):,}; "
                 f"region coverage range {conf.get('region_coverage_min', float('nan')):.0%}–{conf.get('region_coverage_max', float('nan')):.0%}). "
                 "Each fold's width is calibrated using inner OOF residuals from the outer training regions; that fold's held-out rows are used only for the coverage audit. "
                 "These are empirical spatial-validation bands, not formal distribution-free conformal guarantees. **"
                 f"{conf.get('below_interval', 0)} areas** fall below the band.")
    else:
        st.info(conf.get("limitation", "Empirical intervals are unavailable for this run because nested spatial validation could not calibrate them without reusing evaluation residuals."))
    low = df[df["below_expected_90"]].sort_values("capacity_vs_expected")
    show_table(low[["name", "province", "region", "population", "actual_beds_per_1000", "expected_lo90",
                    "expected_beds_per_1000", "gap_index", "priority"]].rename(columns={
        "actual_beds_per_1000": "Reference value per 1,000", "expected_lo90": "Empirical lower band",
        "expected_beds_per_1000": "Cross-fitted estimate per 1,000"}).round(2),
        hide_index=True, use_container_width=True, height=320)
rm = met.get("residual_moran")
if rm and rm.get("I") is not None:
    st.markdown("**Does location matter beyond these characteristics?** (Moran's I of the model's errors)")
    st.write(f"Moran's I = **{rm['I']:.2f}** (p = {rm['p']:.3f}). "
             + ("The model's errors cluster on the map: neighbouring areas are over- or under-supplied together, "
                "so hospital capacity depends on <b>where</b> an area is (regional investment and referral networks), "
                "not only on its people's age, poverty, or isolation." if rm["p"] < 0.05 else
                "The errors show no clear geographic pattern."), unsafe_allow_html=True)

access_f, imp_f = PROC / "access_estimate.json", PROC / "imputation.json"
if access_f.exists():
    access_est = json.loads(access_f.read_text())
    section("Transport assumptions", "Estimated hospital access")
    st.write(
        f"The model estimates **{access_est['population_within_threshold']:,}** residents within "
        f"30 minutes of a hospital/infirmary and **{access_est['population_outside_threshold']:,}** outside that threshold "
        f"({100 * access_est['share_outside_threshold']:.1f}% of the modeled population). These are population-weighted "
        "model estimates—not observed counts of people unable to obtain care."
    )
    ta = access_est["transport_assumptions"]
    st.write(
        f"Road routes use road-class speed assumptions and an access/egress speed of "
        f"{ta['road_access_egress_speed_kmh']:g} km/h. Where a usable route is unavailable, the configured "
        f"effective-speed fallback is {ta['fallback_effective_speed_kmh']:g} km/h with a circuity factor of "
        f"{ta['straight_line_circuity']:g}. These are explicit transport-model assumptions, configured road-network and fallback transport assumptions."
    )
    st.caption(
        "The modeled hospital/infirmary estimate is calculated from the facility inventory and configured transport "
        "assumptions. The supplied DOH 30-minute reference is shown separately as an external comparison; the often-cited "
        "PHFDP figure concerns primary-care centres, so definitions and service types must align before values are compared."
    )
if imp_f.exists():
    imp_r = json.loads(imp_f.read_text())
    section("Model 4 · Graph learning", "Filling the DOH's 'No Data' gaps")
    st.write("Some provinces report no disease data. Instead of assuming a missing province looks like its region's "
             "median, three methods are tested by hiding whole provinces that do report and predicting them: the regional "
             "median, **label propagation** on a graph linking each city or municipality to its 10 nearest neighbours, and "
             "a **graph model** (gradient boosting on the propagated value plus poverty, age, isolation and density). "
             "For each disease, the method with the lowest error on the hidden provinces is used.")
    rows = []
    for c, r in imp_r.items():
        rows.append({"Disease": {"rate_tb": "Tuberculosis", "rate_pneumonia": "Pneumonia",
                                 "rate_hypertension": "Hypertension"}.get(c, c),
                     "Areas with no data": r["n_missing"], "Regional median (error)": round(r["mae_log_regional_median"], 3),
                     "Label propagation (error)": round(r["mae_log_label_propagation"], 3),
                     "Graph model (error)": round(r["mae_log_graph_model"], 3), "Used": r["method_used"].capitalize(),
                     "Change in error vs regional median": f"{-100 * r['error_reduction_vs_median']:+.0f}%"})
    show_table(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Error = mean absolute error of log(1 + rate) on provinces hidden during testing (lower is better). "
               "Where the regional median still wins, reported rates differ between neighbouring provinces mostly "
               "because of how completely each reports cases, which no map-based method can recover.")

tinfo = met.get("typology", {})
section("Model 2 · Clustering", f"{tinfo.get('k', 5)} types of area",
        "K-means clustering on the need and capacity domain scores. Each type suggests a different response, "
        "for example new beds versus better transport versus primary care.")
if tinfo:
    sil = pd.DataFrame({"k": [int(k) for k in tinfo["silhouette"]], "Silhouette": list(tinfo["silhouette"].values())})
    v1, v2 = st.columns([1.2, 1])
    v1.altair_chart(alt.Chart(sil).mark_line(point=True, color=NAVY).encode(
        x=alt.X("k:O", title="Number of types"), y=alt.Y("Silhouette:Q", title="Silhouette score (higher = clearer types)"),
        tooltip=[alt.Tooltip("k:O", title="Number of types"), alt.Tooltip("Silhouette:Q", format=".3f")]).properties(height=200),
        use_container_width=True)
    v2.write(f"The number of types is not fixed in advance: it is the one with the highest **silhouette score** "
             f"(k = {tinfo['k']}). Refitting on 30 bootstrap samples reproduces the same types with an adjusted Rand "
             f"index of **{tinfo['stability_ari']:.2f}** (1 = identical), so the types are "
             f"{'stable' if tinfo['stability_ari'] >= 0.75 else 'only moderately stable'}.")
typ = df.groupby("cluster_name").agg(LGUs=("psgc", "size"), people_M=("population", lambda x: x.sum() / 1e6),
                                     mean_HGI=("gap_index", "mean"), priority_share=("priority", "mean"))
typ["priority_share"] = (typ["priority_share"] * 100).round(0)
show_table(typ.sort_values("mean_HGI", ascending=False).round(2).rename(columns={
    "people_M": "people (M)", "mean_HGI": "mean HGI", "priority_share": "% priority LGUs"}), use_container_width=True)
dom = [c for c in df.columns if c.startswith(("hvs_", "hcs_")) and not c.endswith("_pct")]
prof = df.groupby("cluster_name")[dom].mean().reset_index().melt(id_vars="cluster_name", var_name="domain", value_name="score")
prof["domain"] = prof["domain"].str.replace("hvs_", "Need · ").str.replace("hcs_", "Capacity · ").str.replace("_", " ")
st.altair_chart(alt.Chart(prof).mark_rect().encode(
    x=alt.X("domain:N", title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y("cluster_name:N", title=None),
    color=alt.Color("score:Q", scale=alt.Scale(scheme="blues"), title="Domain score"),
    tooltip=[alt.Tooltip("cluster_name:N", title="Type of area"), alt.Tooltip("domain:N", title="Domain"),
             alt.Tooltip("score:Q", format=".0f", title="Score (0–100)")]).properties(height=260),
    use_container_width=True)
st.caption("Models: mean baseline, LinearRegression, Ridge, RandomForestRegressor (300 trees) and HistGradientBoostingRegressor; "
           "region-grouped outer CV with nested model selection, fold-local imputation/scaling, region-block bootstrap error intervals, "
           "held-out permutation importance, fold-specific training-calibrated empirical bands (not formal conformal guarantees), Moran's I of residuals "
           "(6-neighbour weights, 499 permutations); KMeans with k chosen by silhouette and bootstrap stability (adjusted Rand index).")
footer(meta)
