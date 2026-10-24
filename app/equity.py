"""Equity: is access to care fair across poverty levels, isolation and age structure?"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import show_table, BLUE, CFG, NAVY, RED, footer, page_header, require_data, section, term
from isagiph.equity import concentration_curve, concentration_index, group_table, weighted_quintile

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("Equity", "Is access to care fair?")

w = res["population"]
res = res.assign(beds_pc=res["accessible_beds"] / res["population"] * 1000,
                 pov_q=weighted_quintile(res["poverty_incidence"], w))
qt = group_table(res, res["pov_q"])
q1, q5 = qt.iloc[0], qt.iloc[-1]
ci_beds = concentration_index(res["poverty_incidence"], res["beds_pc"], w)
ci_cov = concentration_index(res["poverty_incidence"], res["cov_any_30"], w)
ci_mal = concentration_index(res["poverty_incidence"], res["cov_malasakit_60"], w)

ratio = q5["accessible beds per 1,000"] / max(q1["accessible beds per 1,000"], 1e-9)
st.markdown(f"""<div class="tiles">
  <div class="tile accent" style="--c:{RED}"><div class="num">{q1['accessible beds per 1,000']:.2f}</div>
    <div class="lbl">hospital beds within reach for every 1,000 people in the poorest 20% of areas</div>
    <div class="note">The least-poor 20% have {q5['accessible beds per 1,000']:.2f}: about {ratio:.1f} times as many.</div></div>
  <div class="tile" style="--c:{RED}"><div class="num">{q1['% ≤30 min to a hospital']:.0f}%</div>
    <div class="lbl">of people in the poorest 20% of areas live within 30 minutes of a hospital</div>
    <div class="note">In the least-poor 20%, {q5['% ≤30 min to a hospital']:.0f}% do.</div></div>
  <div class="tile" style="--c:{BLUE}"><div class="num">{q1['% of people in priority LGUs']:.0f}%</div>
    <div class="lbl">of people in the poorest 20% of areas live in a {term('priority', 'priority')} area</div>
    <div class="note">In the least-poor 20%, {q5['% of people in priority LGUs']:.0f}% do.</div></div>
  <div class="tile" style="--c:{BLUE}"><div class="num">{ci_beds:+.2f}</div>
    <div class="lbl">fairness score for hospital beds (concentration index)</div>
    <div class="note">0 means beds are shared equally between poorer and richer areas; above 0 means richer areas
      get more than their share (the further from 0, the more unequal).</div></div>
</div>""", unsafe_allow_html=True)
st.caption("Areas are grouped into five equal-sized groups of people, from the poorest 20% (highest poverty rate) "
           "to the least-poor 20%.")

section("Poverty", "Who gets the beds?")
c1, c2 = st.columns([1.1, 1])
with c1:
    curves = []
    for lab, h in [("Accessible beds", res["beds_pc"]), ("Within 30 min of a hospital", res["cov_any_30"]),
                   ("Within 60 min of a Malasakit Center", res["cov_malasakit_60"])]:
        cc = concentration_curve(res["poverty_incidence"], h, w).assign(measure=lab)
        curves.append(cc)
    cc = pd.concat(curves)
    eq = pd.DataFrame({"cum_pop": [0, 1], "cum_good": [0, 1]})
    chart = (alt.Chart(cc).mark_line(strokeWidth=3).encode(
                x=alt.X("cum_pop:Q", title="Cumulative share of people (poorest areas first)", axis=alt.Axis(format="%")),
                y=alt.Y("cum_good:Q", title="Cumulative share of the service", axis=alt.Axis(format="%")),
                color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="bottom", columns=1),
                                scale=alt.Scale(range=[RED, NAVY, BLUE])))
             + alt.Chart(eq).mark_line(strokeDash=[5, 5], color="#999").encode(x="cum_pop:Q", y="cum_good:Q"))
    st.altair_chart(chart.properties(height=380), use_container_width=True)
with c2:
    if ci_beds > 0.02:
        msg_beds = "Beds within reach are concentrated in less-poor areas."
    elif ci_beds < -0.02:
        msg_beds = "Beds within reach lean towards poorer areas."
    else:
        msg_beds = "Beds are spread roughly in proportion to population across poverty levels."
    msg_cov = "Fast access favours less-poor areas." if ci_cov > 0.01 else "Fast access is spread fairly evenly."
    msg_mal = ("Malasakit Centers sit closer to less-poor areas, though they exist to help poorer patients."
               if ci_mal > 0.02 else "Malasakit reach is fairly even across poverty levels.")
    st.markdown(f"""
**Reading the numbers**

- **Concentration index for beds: {ci_beds:+.3f}.** {msg_beds}
- **30-minute coverage: {ci_cov:+.3f}.** {msg_cov}
- **Malasakit reach: {ci_mal:+.3f}.** {msg_mal}

The index runs from −1 to +1: 0 is perfect equality, positive values favour less-poor LGUs, and negative values
favour poorer LGUs.
""")

st.markdown("**By poverty fifth** (population-weighted; each fifth holds about 20% of Filipinos). "
            "Q1 = poorest fifth, Q5 = least-poor fifth.")
QCOL = [RED, "#E8913A", "#C9C3B5", "#7FA6D6", NAVY]
qd = qt.reset_index().rename(columns={"_g": "fifth"})
qd["Q"] = ["Q1", "Q2", "Q3", "Q4", "Q5"][: len(qd)]
target, goal = CFG["benchmarks"]["beds_per_1000"], CFG["benchmarks"].get("goal_2040_per_1000", 4.0)


def qbar(col, title, rule=None):
    ch = alt.Chart(qd).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=26).encode(
        x=alt.X("Q:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{col}:Q", title=None),
        color=alt.Color("Q:N", legend=None, scale=alt.Scale(range=QCOL)),
        tooltip=[alt.Tooltip("fifth:N", title="Poverty fifth"), alt.Tooltip(f"{col}:Q", format=".2f", title=title)])
    if rule is not None:
        ch = ch + alt.Chart(pd.DataFrame({"y": [rule]})).mark_rule(color=RED, strokeDash=[5, 4], size=2).encode(y="y:Q")
    return ch.properties(height=240, title=alt.TitleParams(title, fontSize=13, anchor="start"))


cq = st.columns(4)
cq[0].altair_chart(qbar("accessible beds per 1,000", "Beds within reach per 1,000", rule=target), use_container_width=True)
cq[1].altair_chart(qbar("% ≤30 min to a hospital", "% within 30 min of a hospital"), use_container_width=True)
cq[2].altair_chart(qbar("% ≤60 min to Malasakit", "% within 60 min of Malasakit"), use_container_width=True)
cq[3].altair_chart(qbar("% of people in priority LGUs", "% living in priority LGUs"), use_container_width=True)
st.caption(f"Dashed red line = DOH target of {target:g} beds per 1,000 (1 per 800 people). Even the least-poor fifth "
           f"is far below the 2040 goal of {goal:g} per 1,000, so a taller bar means more beds, not enough beds.")
show_table(qt.round(2).rename_axis("Poverty fifth"), use_container_width=True)

section("Isolation", "Geographically isolated communities")
gida = np.where(res["gida_share"] >= 0.5, "Mostly GIDA (≥50% of barangays)",
                np.where(res["gida_share"] > 0, "Some GIDA barangays", "No GIDA barangays"))
show_table(group_table(res, pd.Series(gida, index=res.index)).round(2), use_container_width=True)

section("Age", "Young and ageing populations")
a1, a2 = st.columns(2)
with a1:
    yq = weighted_quintile(res["pct_age_0_4"], w, labels=("Most children", "Q2", "Q3", "Q4", "Fewest children"))
    st.markdown("**By share of children aged 0–4**")
    show_table(group_table(res, yq)[["people (M)", "accessible beds per 1,000", "% ≤30 min to a hospital",
                                       "% of people in priority LGUs"]].round(2), use_container_width=True)
with a2:
    oq = weighted_quintile(res["pct_age_60_plus"], w, labels=("Most older adults", "Q2", "Q3", "Q4", "Fewest older adults"))
    st.markdown("**By share of adults aged 60+**")
    show_table(group_table(res, oq)[["people (M)", "accessible beds per 1,000", "% ≤30 min to a hospital",
                                       "% of people in priority LGUs"]].round(2), use_container_width=True)

section("Regions", "No region has enough beds")
SHORT = {"National Capital Region (NCR)": "NCR", "Cordillera Administrative Region (CAR)": "CAR",
         "Region I (Ilocos Region)": "Ilocos (I)", "Region II (Cagayan Valley)": "Cagayan Valley (II)",
         "Region III (Central Luzon)": "Central Luzon (III)", "Region IV-A (Calabarzon)": "CALABARZON (IV-A)",
         "Mimaropa Region": "MIMAROPA", "Region V (Bicol Region)": "Bicol (V)", "Region VI (Western Visayas)": "W. Visayas (VI)",
         "Region VII (Central Visayas)": "C. Visayas (VII)", "Region VIII (Eastern Visayas)": "E. Visayas (VIII)",
         "Region IX (Zamboanga Peninsula)": "Zamboanga (IX)", "Region X (Northern Mindanao)": "N. Mindanao (X)",
         "Region XI (Davao Region)": "Davao (XI)", "Region XII (Soccsksargen)": "SOCCSKSARGEN (XII)",
         "Region XIII (Caraga)": "Caraga (XIII)",
         "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)": "BARMM"}
rg = res.groupby("region").agg(pop=("population", "sum"), beds=("accessible_beds", "sum")).reset_index()
rg["name"] = rg["region"].map(lambda x: SHORT.get(x, x))
rg["per1000"] = 1000 * rg["beds"] / rg["pop"]
rg["pct_goal"] = 100 * rg["per1000"] / goal
order = list(rg.sort_values("per1000", ascending=False)["name"])
national = 1000 * rg["beds"].sum() / rg["pop"].sum()
r1, r2 = st.columns(2)
with r1:
    bars = alt.Chart(rg).mark_bar(cornerRadiusEnd=4, size=14, color=NAVY).encode(
        y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("per1000:Q", title="Beds within reach per 1,000 people", scale=alt.Scale(domain=[0, goal + 0.3])),
        tooltip=[alt.Tooltip("region:N", title="Region"), alt.Tooltip("per1000:Q", format=".2f", title="Beds per 1,000"),
                 alt.Tooltip("pct_goal:Q", format=".0f", title="% of the 2040 goal")])
    rules = alt.Chart(pd.DataFrame({"x": [target, goal], "label": [f"DOH target {target:g}", f"2040 goal {goal:g}"]}))
    lines = rules.mark_rule(strokeDash=[5, 4], size=2).encode(
        x="x:Q", color=alt.Color("label:N", title=None, scale=alt.Scale(range=[RED, "#E8913A"]),
                                 legend=alt.Legend(orient="top", labelLimit=0)))
    st.altair_chart((bars + lines).properties(height=520), use_container_width=True)
with r2:
    sh = rg.assign(Population=100 * rg["pop"] / rg["pop"].sum(), Beds=100 * rg["beds"] / rg["beds"].sum())
    sh = sh.melt(id_vars=["name", "region"], value_vars=["Population", "Beds"], var_name="share of", value_name="%")
    sh["share of"] = sh["share of"].map({"Population": "Share of population", "Beds": "Share of beds within reach"})
    st.altair_chart(alt.Chart(sh).mark_bar(cornerRadiusEnd=3, size=7).encode(
        y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)), x=alt.X("%:Q", title="% of national total"),
        color=alt.Color("share of:N", title=None, scale=alt.Scale(range=[RED, NAVY]), legend=alt.Legend(orient="top", labelLimit=0)),
        yOffset="share of:N", tooltip=[alt.Tooltip("region:N", title="Region"), alt.Tooltip("share of:N", title="Measure"),
                                   alt.Tooltip("%:Q", format=".1f", title="% of national total")]).properties(height=520),
        use_container_width=True)
best = rg.sort_values("per1000", ascending=False).iloc[0]
st.info(f"Nationally there are **{national:.2f} beds within reach per 1,000 people**, below the DOH target of {target:g}. "
        f"The best-supplied region, {best['name']}, has {best['per1000']:.2f} per 1,000, only "
        f"{best['pct_goal']:.0f}% of the 2040 goal of {goal:g}. A region with a larger share of beds than of people is "
        "better supplied than the others, not adequately supplied.")
st.caption("Equity statistics use LGU averages, so inequality inside each LGU is not visible. Poverty for highly "
           "urbanised cities is imputed with the regional median.")
footer(meta)
