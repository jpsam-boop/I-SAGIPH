"""Plain-language methods guide plus the equations and diagnostics used by I-SAGIPH."""
import streamlit as st

from common import INDICATOR_LABELS, NAVY, ROOT, footer, page_header, read_text, require_data

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("Learn · Methods", "How it works",
            "A guided explanation of the data, statistics, machine-learning models, and spatial calculations behind I-SAGIPH.")

st.markdown("""
I-SAGIPH brings public population, poverty, disease, facility, geography, and transport information into a common LGU-level analysis. It asks three related questions: **where is healthcare need high, how much care can people plausibly reach, and where does need exceed reachable capacity?** The outputs help identify areas for investigation; they are not individual medical advice, live facility availability, or automatic funding decisions.
""")

st.markdown("### The analysis, from data to decision support")
with st.container(border=True):
    st.markdown("""
1. **Prepare and link records.** Standardise LGU identifiers and geographic boundaries, consolidate facility lists, identify missing or imputed values, and document data quality.
2. **Estimate travel time and reachable care.** Use mapped roads and configured road-class speed assumptions where route data are available; use the documented generalized fallback when a usable route is unavailable. Assumptions reflect a mixed local transport environment, but do not individually route every jeepney, tricycle, walking leg, transfer, wait, fare, or ferry crossing.
3. **Calculate accessibility.** Estimate how beds are shared among populations that can reach facilities using a Two-Step Floating Catchment Area (E2SFCA) model.
4. **Build comparable indices.** Combine need indicators and capacity indicators into 0–100 relative scores, then compare their percentiles to show the healthcare gap.
5. **Test patterns and robustness.** Use spatial statistics, model validation, bootstrap intervals, and alternative index assumptions to show whether results vary by geography or modeling choice.
6. **Explore practical scenarios.** The ISAGIP Simulator and Crisis Impact tools recalculate modeled indicators under hypothetical additions, closures, or capacity changes.
""")

t1, t2, t3, t4, t5 = st.tabs(["Models & formulas", "Statistics & ML", "Weights & indices", "Data & limitations", "Sources"])

with t1:
    st.markdown("#### 1. Travel-time accessibility (E2SFCA)")
    st.write("E2SFCA estimates how much facility capacity is potentially available to residents after accounting for the other populations who can reach the same facility. Closer facilities receive higher travel-time weights. It is an accessibility model, not a count of patients who actually travel or receive treatment.")
    st.latex(r"R_j=\frac{S_j}{\sum_k P_k W(d_{kj})},\qquad A_i=\sum_j R_j W(d_{ij})")
    st.write("Here, S is facility supply (for example, licensed beds), P is population demand, d is modeled travel time, R is supply relative to catchment demand, and A is the accessibility estimate for a demand location. Demand-point results are aggregated to the city/municipality level.")
    st.markdown("#### 2. Transport assumptions")
    st.write("Road-network routes use a configured speed profile by road class plus a short access/egress leg between a population/facility point and a mapped road. If a usable route is unavailable, the model estimates time from straight-line distance using a circuity factor and a generalized effective-speed assumption. This approximates mixed local travel conditions but is not person-specific door-to-door routing. The model's estimate of the population beyond 30 minutes from a hospital is compared with the supplied DOH reference figure as an external check.")
    st.markdown("#### 3. Consolidating indicators")
    st.write("Indicators are scaled to comparable ranges. Principal-component/factor analysis identifies shared patterns; factors are varimax-rotated, indicator weights are based on squared loadings, and factor weights reflect explained variance. Weighted geometric aggregation means a very weak domain is not completely offset by a strong one.")
    st.markdown("#### 4. The healthcare gap")
    st.latex(r"HGI_i=\operatorname{percentile}(HNI_i)-\operatorname{percentile}(HCI_i)")
    st.write("HNI summarises relative need; HCI summarises relative capacity within reach; HGI subtracts capacity rank from need rank. Positive HGI means need ranks ahead of capacity. These are relative comparisons across LGUs, while beds per 1,000 and the share of the 2040 goal provide separate adequacy measures.")
    st.markdown("#### 5. Need-capacity groups and priority stability")
    st.write("A 3 × 3 grid combines high, middle, or low need with low, middle, or high capacity. The priority-robustness procedure repeats the index calculation under alternative defensible weights, scaling choices, and catchment sizes. Priority frequency is the share of runs in which an LGU is classified as a priority area; it measures sensitivity to these choices, not a probability that the area deserves funding.")

with t2:
    st.markdown("#### Regression and model validation")
    st.write("The Regression Explorer predicts the E2SFCA-derived accessible-bed indicator from demographic and socioeconomic characteristics. It compares a mean baseline, linear regression, Ridge regression, Random Forest, and gradient boosting. Region-grouped folds hold out whole geographic groups when enough groups are available; imputation and scaling are learned inside training folds, and nested model selection reduces optimism from choosing among models.")
    st.markdown("- **MAE:** typical absolute error in beds per 1,000; smaller is better.\n- **RMSE:** gives large errors extra weight; smaller is better.\n- **R²:** how closely predictions track variation in the derived reference; it can be negative or undefined.\n- **Bias:** average signed prediction error; a positive value means overestimation.\n- **Permutation importance:** measures how much held-out predictive performance changes when one feature is shuffled. It is not a causal effect or a complete explanation of one LGU.")
    st.write("Prediction bands are empirical spatial-validation diagnostics estimated from training-fold cross-fitted residuals and assessed on held-out regions. They are not guaranteed prediction intervals for every LGU.")
    st.markdown("#### Relationship and equity statistics")
    st.write("The Health Relationships workspace uses Spearman's rank correlation to describe whether two variables tend to move together, a fitted linear trend to show the overall direction, and a region-block bootstrap for uncertainty where enough groups are available. An optional multivariable linear model adjusts for selected measured covariates and uses region-clustered inference when there are enough regions. These remain observational associations, not proof of causation.")
    st.write("Equity analysis uses population-weighted poverty fifths, concentration curves and concentration indices to show how accessible beds, travel-time coverage, and assistance reach are distributed across poorer and less-poor LGUs. Values near zero imply a more proportional distribution; positive values indicate concentration among less-poor areas under this convention. Results describe distribution rather than explaining its cause.")
    st.markdown("#### Spatial statistics and missing-data estimation")
    st.write("Moran's I tests whether similar gap values cluster spatially; LISA and Getis–Ord Gi* identify local patterns, with false-discovery-rate correction for multiple local tests. Graph-based imputation compares regional-median, label-propagation and graph-model estimates by hiding known provinces during evaluation before applying a method to missing records. The selected method depends on held-out error; imputed values remain estimates, not observed disease counts.")
    st.markdown("#### Facility optimization and disruption")
    st.write("The simulator's suggested-site routine uses a greedy, need-weighted maximal-covering strategy to choose candidate locations. Crisis Impact removes a facility's supply or reduces licensed beds, then recalculates accessibility and indices. Both are conditional spatial calculations; they do not simulate actual patient choices, construction, staffing, queues, transport service disruptions, or referral capacity.")

with t3:
    st.markdown("#### What the indices mean")
    st.markdown("- **Health Need Index (HNI), 0–100:** relative need based on age, poverty, GIDA isolation and selected disease indicators.\n- **Hospital Capacity Index (HCI), 0–100:** relative reachable capacity, including accessible beds, coverage, primary-care availability and Malasakit reach.\n- **Healthcare Gap Index (HGI), −100 to +100:** need percentile minus capacity percentile. Positive values show need outranking capacity.\n- **Bed need met:** reachable beds per 1,000 as a share of the 2040 planning goal of 4 beds per 1,000. This is not a percentile ranking.\n- **Bed shortfall:** estimated number of additional beds needed under the configured population and benchmark assumptions; licensed capacity is not the same as staffed or operational beds.")
    if len(weights):
        wt = weights.assign(Indicator=weights["indicator"].map(lambda c: INDICATOR_LABELS.get(c, c)), Weight=(weights["overall_weight"] * 100).round(1))
        for idx_name, group in wt.groupby("index", sort=False):
            st.markdown(f"**{idx_name}** · {group['variance_explained'].iloc[0]:.0%} of variance explained by {group['factor'].nunique()} factor(s)")
            st.bar_chart(group.set_index("Indicator")["Weight"], horizontal=True, height=max(130, 34 * len(group)), color=NAVY)
            st.caption("A larger weight means the indicator contributes more to this configured composite score. It does not mean that the factor causes access gaps.")
    else:
        st.info("This run used equal weights. Rebuild the analysis with the source indicators to calculate factor-based weights.")

with t4:
    st.markdown("#### Data sources and coverage")
    st.markdown("""
| Source family | How it is used |
|---|---|
| PSA census and Small Area Estimates | Population, age structure and poverty |
| DOH facility and health statistics | Licensed facility inventory, selected morbidity indicators, GIDA classification, and the supplied external access reference |
| PCSO Malasakit listings | Indication of geographic reach to listed assistance centres; lists may be dated |
| OCHA / PSA / NAMRIA boundaries | LGU and barangay outlines or representative points |
| OpenStreetMap | Mapped facility and road-network geometry where available |
""")
    road_share = meta.get("road_network_population_share")
    if road_share is not None:
        st.info(f"In this run, {float(road_share):.1%} of population has road-network travel-time coverage. Areas relying on fallback estimates have different travel-time evidence; compare them cautiously.")
    st.markdown("#### What the analysis cannot tell you")
    st.markdown("- Model travel time is not a measurement of every Filipino's real journey. Public transport timetables, transfers, fares, waiting, and every boat/ferry connection are not individually represented.\n- Licensed beds are not the same as staffed beds, vacancies, or usable service capacity. Private-care affordability is not modeled.\n- Demand is allocated to representative points; people within the same LGU can have very different journeys.\n- Municipal disease data may inherit province-level reported rates; missing values may be imputed, and reported rates depend on reporting completeness.\n- The regression target is derived from the same accessibility methodology. It is not independent validation against observed patient use, clinical outcomes, or actual staffed capacity.\n- Results vary with source vintage, missing data, facility coordinates, route coverage, and the transport and index assumptions. Refresh source files and inspect the data-quality report before formal use.")
    dq = read_text(ROOT / "reports" / "data_quality.md")
    if dq:
        with st.expander("Current data-linking audit", expanded=False):
            st.markdown(dq)

with t5:
    st.markdown(read_text(ROOT / "docs" / "references.md") or "See docs/references.md.")

footer(meta)
