"""Interactive LGU-level explorer for the existing spatially cross-fitted regression outputs."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import BLUE, NAVY, PROC, RED, footer, page_header, require_data, section, show_table
from isagiph.analysis_tools import regression_metrics

res, fac, meta, valid, weights, gj, pts = require_data()
page_header(
    "Data Models · Interactive",
    "Regression Explorer",
    "Investigate how the model estimates E2SFCA-derived accessible beds per 1,000 residents across LGUs. "
    "Select a geography and model, compare held-out predictions with the derived reference values, and inspect local errors.",
)

st.info("**What this means:** closer agreement between held-out predictions and the reference suggests the model can reproduce this derived accessibility measure for unseen regions. Large errors show where estimates should be treated cautiously; this is not proof of real-world clinical accuracy.")

pred_path, metric_path = PROC / "ml_predictions.csv", PROC / "ml_metrics.json"
if not pred_path.exists() or not metric_path.exists():
    st.info("Regression outputs are not available yet. Run `python -m isagiph.pipeline` (or `--demo`) to generate model predictions.")
    st.stop()

pred = pd.read_csv(pred_path, dtype={"psgc": str})
metrics = json.loads(metric_path.read_text(encoding="utf-8"))
imp_path = PROC / "ml_importance.csv"
importance = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
df = res.copy()
df["psgc"] = df["psgc"].astype(str)
df = df.merge(pred, on="psgc", how="inner", suffixes=("", "_prediction"))

model_options = [("Nested model selection (cross-fitted)", "expected_beds_per_1000")]
for key, label in (("mean_baseline", "Mean baseline"), ("linear", "Linear regression"),
                   ("ridge", "Ridge regression"), ("random_forest", "Random forest"),
                   ("gradient_boosting", "Gradient boosting")):
    col = f"cv_pred_{key}_beds_per_1000"
    if col in df.columns:
        model_options.append((label, col))
if len(df) == 0 or "actual_beds_per_1000" not in df.columns:
    st.warning("No matching LGU-level cross-fitted prediction rows were found. Re-run the ML pipeline to rebuild outputs.")
    st.stop()

section("Explore", "Choose what to investigate")
f1, f2, f3 = st.columns([1.05, 1.5, 1.1])
regions = ["All regions"] + sorted(df["region"].dropna().astype(str).unique().tolist()) if "region" in df else ["All regions"]
region = f1.selectbox("Geographic scope", regions, key="regexpl_region")
model_label = f2.selectbox("Prediction model", [m[0] for m in model_options], key="regexpl_model")
chosen_col = dict(model_options)[model_label]
error_filter = f3.selectbox("Focus", ["All LGUs", "Largest overestimates", "Largest underestimates", "Largest absolute errors"], key="regexpl_focus")
scope_view = df.copy()
if region != "All regions":
    scope_view = scope_view[scope_view["region"].astype(str) == region]
scope_view = scope_view.replace([np.inf, -np.inf], np.nan)
view = scope_view.copy()
view["selected_prediction"] = pd.to_numeric(view[chosen_col], errors="coerce")
view["prediction_error"] = view["selected_prediction"] - pd.to_numeric(view["actual_beds_per_1000"], errors="coerce")
view["absolute_error"] = view["prediction_error"].abs()
view = view.dropna(subset=["selected_prediction", "actual_beds_per_1000"]).copy()
if error_filter == "Largest overestimates":
    view = view.sort_values("prediction_error", ascending=False).head(200)
elif error_filter == "Largest underestimates":
    view = view.sort_values("prediction_error", ascending=True).head(200)
elif error_filter == "Largest absolute errors":
    view = view.sort_values("absolute_error", ascending=False).head(200)

if view.empty:
    st.warning("No usable predictions match these filters.")
    st.stop()

# Headline metrics always use every eligible LGU in the selected geographic scope.
# The error-focus filter only changes the records displayed for investigation; it must
# never make an error-selected subset look like an unbiased performance evaluation.
overall_view = scope_view.copy().replace([np.inf, -np.inf], np.nan)
overall_view["selected_prediction"] = pd.to_numeric(overall_view[chosen_col], errors="coerce")
overall_view["prediction_error"] = overall_view["selected_prediction"] - pd.to_numeric(overall_view["actual_beds_per_1000"], errors="coerce")
overall_view = overall_view.dropna(subset=["selected_prediction", "actual_beds_per_1000"]).copy()
score = regression_metrics(overall_view["actual_beds_per_1000"], overall_view["selected_prediction"])
k1, k2, k3, k4 = st.columns(4)
k1.metric("LGUs in geographic scope", f"{score['n']:,}")
k2.metric("Overall MAE", f"{score['mae']:.3f}" if np.isfinite(score["mae"]) else "N/A", help="Mean absolute difference in beds per 1,000 across every eligible LGU in the selected geographic scope, independent of the focus filter.")
k3.metric("Overall RMSE", f"{score['rmse']:.3f}" if np.isfinite(score["rmse"]) else "N/A", help="Penalises larger errors more heavily; calculated on all eligible LGUs in the selected geographic scope.")
k4.metric("Overall R²", f"{score['r2']:.3f}" if np.isfinite(score["r2"]) else "N/A", help="R² is computed on the original accessibility-indicator scale. It may be undefined when the reference values have little variation.")
if error_filter != "All LGUs":
    st.info(f"Error investigation filter: **{error_filter}**. The plot and LGU selector below focus on {len(view):,} records; the overall metrics above remain calculated over all {len(overall_view):,} eligible LGUs in this geographic scope.")
r2_value = score.get("r2", np.nan)
if np.isfinite(r2_value) and r2_value < 0:
    r2_interpretation = "R² is below zero in this scope, meaning the predictions perform worse than using the scope's mean reference value as a predictor."
elif np.isfinite(r2_value):
    r2_interpretation = f"R² is {r2_value:.2f}; it summarises how much variation in the derived reference the predictions track, and is not an accuracy percentage."
else:
    r2_interpretation = "R² is unavailable because the reference values or predictions do not support a stable calculation."
st.info("**How to read these metrics:** MAE is the typical absolute miss in beds per 1,000 (lower is closer); RMSE penalises large misses more heavily; bias shows systematic over- or underestimation. " + r2_interpretation + " If errors are large in particular areas, review their input coverage and travel-time evidence before using the estimate for planning.")
st.caption("Predictions are cross-fitted / out-of-fold. The reference is an E2SFCA-derived accessibility indicator—not observed staffed beds, patient utilisation, service quality, or a causal outcome. Error = prediction − reference; positive means overestimate. Metrics on this page are predictive agreement with that derived indicator, not validation against independently observed healthcare access.")

section("Compare", "Reference values vs model predictions")
plot = view.copy()
plot["priority_label"] = plot["priority"].map({True: "Priority", False: "Other"}) if "priority" in plot else "LGU"
maxv = float(max(plot["actual_beds_per_1000"].max(), plot["selected_prediction"].max(), 0.1))
minv = 0.0
ideal = pd.DataFrame({"reference": [minv, maxv], "prediction": [minv, maxv]})
base = alt.Chart(plot).encode(
    x=alt.X("actual_beds_per_1000:Q", title="Derived accessibility reference · beds per 1,000", scale=alt.Scale(domain=[minv, maxv])),
    y=alt.Y("selected_prediction:Q", title=f"{model_label} prediction · beds per 1,000", scale=alt.Scale(domain=[minv, maxv])),
    tooltip=[alt.Tooltip("name:N", title="LGU"), alt.Tooltip("province:N", title="Province"),
             alt.Tooltip("region:N", title="Region"), alt.Tooltip("actual_beds_per_1000:Q", format=".3f", title="Reference"),
             alt.Tooltip("selected_prediction:Q", format=".3f", title="Prediction"),
             alt.Tooltip("prediction_error:Q", format="+.3f", title="Prediction − reference")]
)
reference_line = alt.Chart(ideal).mark_line(color="#7B8798", strokeDash=[5, 4], strokeWidth=2).encode(
    x=alt.X("reference:Q"), y=alt.Y("prediction:Q"))
scatter = base.mark_circle(size=62, opacity=0.72).encode(
    color=alt.Color("prediction_error:Q", title="Prediction error", scale=alt.Scale(scheme="redblue", domainMid=0)),
    size=alt.Size("population:Q", title="Population", scale=alt.Scale(range=[24, 220]), legend=None),
).properties(height=440).interactive()
st.altair_chart(reference_line + scatter, use_container_width=True)
st.caption("The dashed diagonal represents perfect agreement. Point size represents LGU population; colour indicates over- or underestimation. The view is filtered to at most 200 records when an error-focused view is selected.")

section("Investigate", "Inspect an individual LGU")
lookup = view.copy()
if "name" not in lookup:
    lookup["name"] = lookup["psgc"]
lookup["_label"] = lookup["name"].astype(str) + " — " + lookup.get("province", pd.Series("", index=lookup.index)).fillna("").astype(str) + " · " + lookup["psgc"].astype(str)
lookup = lookup.sort_values("_label")
default_row = lookup["absolute_error"].idxmax()
labels = lookup["_label"].tolist()
default_label = lookup.loc[default_row, "_label"]
selected_label = st.selectbox("Select an LGU", labels, index=labels.index(default_label), key="regexpl_lgu")
r = lookup.loc[lookup["_label"] == selected_label].iloc[0]
a, b, c, d = st.columns(4)
a.metric("Reference value", f"{r['actual_beds_per_1000']:.3f}")
b.metric("Selected prediction", f"{r['selected_prediction']:.3f}")
c.metric("Prediction error", f"{r['prediction_error']:+.3f}")
d.metric("Model assigned during cross-fitting", str(r.get("model_used", "not recorded")).replace("_", " ").title())

left, right = st.columns([1.15, 1])
with left:
    with st.container(border=True):
        st.markdown(f"### {r.get('name', 'Selected LGU')}")
        st.write(f"{r.get('province', '')}, {r.get('region', '')}")
        features = [("Population", "population", "{:,.0f}"), ("Poverty incidence", "poverty_incidence", "{:.1%}"),
                    ("GIDA share", "gida_share", "{:.1%}"), ("Age 0–4 share", "pct_age_0_4", "{:.1%}"),
                    ("Age 60+ share", "pct_age_60_plus", "{:.1%}")]
        rows = []
        for label, col, fmt in features:
            val = r.get(col, np.nan)
            try:
                valid_val = pd.notna(val) and np.isfinite(float(val))
            except (TypeError, ValueError):
                valid_val = False
            if valid_val:
                value = fmt.format(float(val))
                rows.append({"Input characteristic": label, "Value": value})
        if rows:
            show_table(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("These are model inputs or contextual characteristics, not causal explanations for the prediction.")
with right:
    with st.container(border=True):
        st.markdown("### Prediction interval")
        band_info = metrics.get("prediction_band", {})
        if (chosen_col == "expected_beds_per_1000" and {"expected_lo90", "expected_hi90"}.issubset(r.index)
                and pd.notna(r.get("expected_lo90")) and pd.notna(r.get("expected_hi90"))):
            nominal = float(band_info.get("level", 0.90)) * 100
            st.write(f"Training-fold-calibrated empirical {nominal:.0f}% band: **{r['expected_lo90']:.3f}–{r['expected_hi90']:.3f}** beds per 1,000.")
            st.caption("For each outer validation fold, the width is calibrated using inner out-of-fold residuals from its training regions; the held-out LGU does not set its own band. Observed coverage is assessed on outer-held-out rows. These remain empirical spatial-validation bands, not formal conformal guarantees.")
            if pd.notna(band_info.get("empirical_coverage_on_outer_heldout_rows", np.nan)):
                st.write(f"Held-out coverage across rows with intervals: **{band_info['empirical_coverage_on_outer_heldout_rows']:.1%}** (n={band_info.get('n_outer_heldout_rows_with_interval', 0):,}).")
                st.caption("National held-out coverage is not a guarantee for this LGU or its region. Review the per-region coverage range in the run's ml_metrics.json.")
        elif chosen_col == "expected_beds_per_1000":
            st.info("An uncertainty band is unavailable for this run. " + str(band_info.get("limitation", metrics.get("interval_limitation", "Nested spatial validation did not provide enough independent training information."))))
        else:
            st.info("A fold-calibrated uncertainty band is available only for the nested cross-fitted selected-model prediction. Candidate-model intervals are not fabricated.")
        fold = r.get("validation_fold", np.nan)
        if pd.notna(fold):
            st.write(f"Held-out validation fold: **{int(fold)}**")
        if "priority" in r and pd.notna(r["priority"]):
            st.write("Current index classification: " + ("priority area" if bool(r["priority"]) else "not classified as a priority area") + ".")

section("Context", "How do candidate models compare in this geographic scope?", "Comparison uses all LGUs in the selected geographic scope, regardless of the error-focused view above.")
model_rows = []
metric_cols = []
for label, col in model_options:
    # Compare models over the full selected geographic scope, not the error-focused
    # subset used to inspect failures above.
    comp = scope_view.dropna(subset=["actual_beds_per_1000", col]) if col in scope_view else pd.DataFrame()
    sc = regression_metrics(comp["actual_beds_per_1000"], comp[col]) if not comp.empty else regression_metrics([], [])
    model_rows.append({"Model": label, "LGUs": sc["n"], "MAE": sc["mae"], "RMSE": sc["rmse"], "R²": sc["r2"], "Mean bias": sc["bias"]})
model_table = pd.DataFrame(model_rows).round({"MAE": 4, "RMSE": 4, "R²": 4, "Mean bias": 4})
show_table(model_table, hide_index=True, use_container_width=True)
st.caption("Candidate predictions are computed on the shared outer spatial folds; they are useful for comparison, but choosing the best-looking candidate after inspecting these scores can create selection optimism. Prefer nested-selection performance in the Data Models page for a less biased estimate of the selection procedure.")

if not importance.empty and {"label", "rf_importance"}.issubset(importance.columns):
    section("Global context", "Overall feature importance")
    imp = importance.sort_values("rf_importance", ascending=False).head(8)
    st.altair_chart(alt.Chart(imp).mark_bar(color=NAVY, cornerRadiusEnd=4).encode(
        x=alt.X("rf_importance:Q", title="Mean held-out permutation importance"),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("rf_importance:Q", format=".4f", title="Importance")]
    ).properties(height=max(180, 28 * len(imp))), use_container_width=True)
    st.caption("This is global model importance averaged across held-out folds. It is not a row-specific causal explanation for the selected LGU.")
footer(meta)
