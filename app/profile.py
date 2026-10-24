"""LGU profile: three index cards, plain-language reading, indicators and comparison."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (cascade, term, show_table, BLUE, DOMAIN_LABELS, HCI, HGI, HNI, INDICATOR_LABELS, MUTED, NAVY, NEED_INDICATORS, RED,
                    footer, page_header, pct_of, require_data)

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("LGU profile", "Look up any city or municipality")

res_lbl = res.assign(label=res["name"] + ", " + res["province"]).sort_values("label")
stable_lbl = res_lbl[res_lbl["priority"] & ~res_lbl["unstable"]].sort_values("gap_index", ascending=False)
default = stable_lbl["label"].iloc[0] if len(stable_lbl) else res_lbl["label"].iloc[0]
labels = res_lbl["label"].tolist()
d0 = res_lbl[res_lbl["label"] == default].iloc[0]
if "profile_pick" not in st.session_state:          # first visit: start on the top stable priority LGU
    st.session_state.profile_pick = default
    st.session_state.prof_reg, st.session_state.prof_prov, st.session_state.prof_lgu = d0["region"], d0["province"], d0["name"]
fragment = getattr(st, "fragment", lambda f: f)


@fragment
def chooser() -> None:
    """Only the dropdowns update while choosing; the profile reloads when Show profile is pressed."""
    with st.container(border=True):
        cc = st.columns([1, 1, 1, 0.6])
        _reg, _prov, _lgu = cascade(cc[:3], sorted(res["region"].unique()), res, "prof", with_lgu=True, multi=False)
        cc[3].markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if cc[3].button("Show profile", type="primary", use_container_width=True):
            st.session_state.profile_pick = f"{_lgu}, {_prov}"
            st.rerun()


chooser()
pick = st.session_state.profile_pick
r = res_lbl[res_lbl["label"] == pick].iloc[0]


def card(title, value, pct, color, sub):
    return (f"<div class='idx-card' style='--c:{color}'><div class='t'>{title}</div><div class='v'>{value}</div>"
            f"<div class='bar'><div style='width:{max(0, min(100, pct)):.0f}%'></div></div>"
            f"<div style='color:{MUTED};font-size:.78rem;margin-top:6px'>{sub}</div></div>")


from isagiph.profile_pdf import lgu_profile_pdf  # noqa: E402
from common import ASSETS, CFG  # noqa: E402

st.download_button("⬇ Download this profile (PDF)",
                   data=lgu_profile_pdf(r, res, ASSETS / "logo_icon.png", CFG["benchmarks"]["beds_per_1000"]),
                   file_name=f"I-SAGIPH profile - {r['name']}, {r['province']}.pdf", mime="application/pdf")

cA, cB, cC = st.columns(3)
cA.markdown(card(HGI, f"{r['gap_index']:+.0f}", (r["gap_index"] + 100) / 2, "#7A4BB0",
                 f"{term('Priority area', 'priority') + ' · ' if r['priority'] else ''}{term('priority', 'priority')} in "
                 f"{100 * r['priority_freq']:.0f}% of runs"),
            unsafe_allow_html=True)
cB.markdown(card(HNI, f"{r['HVS']:.0f}", r["hvs_pct"], "#d6604d", f"Higher need than {r['hvs_pct']:.0f}% of areas"),
            unsafe_allow_html=True)
cC.markdown(card(HCI, f"{r['HCS']:.0f}", r["hcs_pct"], BLUE, f"More than {r['hcs_pct']:.0f}% of areas · {r['beds_per_1000_all']:.2f} beds per 1,000 "
                 f"({100 * r['beds_per_1000_all'] / 4:.0f}% of the 2040 goal)"),
            unsafe_allow_html=True)

need_ind = [c for c in INDICATOR_LABELS if c in res.columns and c in NEED_INDICATORS]
cap_ind = [c for c in INDICATOR_LABELS if c in res.columns and c not in need_ind]
need_drivers = sorted(need_ind, key=lambda c: -pct_of(res[c], r[c]))[:2]
cap_gaps = sorted(cap_ind, key=lambda c: pct_of(res[c], r[c]))[:2]
tier = {3: "top third", 2: "middle third", 1: "bottom third"}
robust = (f"This is a {term('stable', 'stable')} result: it held across most model variations." if not r["unstable"]
          else f"This is an {term('unstable', 'unstable')} result: it depends on modelling choices, so verify locally "
               "before acting.")
st.markdown(
    f"**Reading this profile.** {r['name']} ({r['province']}) is in the **{tier[r['v_tertile']]}** of areas for health "
    f"need and the **{tier[r['c_tertile']]}** for hospital capacity (compared with other areas; it has "
    f"{r['beds_per_1000_all']:.2f} beds within reach per 1,000 people against the DOH target of 1.25 and the 2040 goal of 4)"
    f"{' — a ' + term('priority area', 'priority') + ' (high need, low capacity)' if r['priority'] else ''}. Need is driven most by "
    f"**{INDICATOR_LABELS[need_drivers[0]].lower()}** ({pct_of(res[need_drivers[0]], r[need_drivers[0]]):.0f}th percentile)"
    + (f" and **{INDICATOR_LABELS[need_drivers[1]].lower()}**" if len(need_drivers) > 1 else "")
    + f". Its weakest capacity measures are **{INDICATOR_LABELS[cap_gaps[0]].lower()}** "
    f"({pct_of(res[cap_gaps[0]], r[cap_gaps[0]]):.0f}th percentile)"
    + (f" and **{INDICATOR_LABELS[cap_gaps[1]].lower()}**" if len(cap_gaps) > 1 else "") + f". {robust}",
    unsafe_allow_html=True)

k1, k2 = st.columns(2)
with k1:
    rows = [("Population (2020)", f"{r['population']:,.0f}")] + [
        (INDICATOR_LABELS[c], f"{r[c]:.1%}" if c.startswith(("pct_", "cov_")) or c in ("poverty_incidence", "gida_share")
         else f"{r[c]:,.2f}") for c in need_ind + cap_ind]
    def kmf(v):
        return "—" if pd.isna(v) else ("<1 km" if v < 1 else f"{v:,.0f} km")
    rows += [("Nearest hospital (straight line)", kmf(r.get("km_hospital"))),
             ("Nearest primary care facility", kmf(r.get("km_primary"))),
             ("Nearest Malasakit Center", kmf(r.get("km_malasakit"))),
             ("Nearest dialysis", kmf(r.get("km_dialysis"))),
             ("Ambulances in this area", f"{r.get('ambulances', 0):.0f} ({r.get('ambulances_per_100k', 0):.1f} per 100k)"),
             ("Hospitals · licensed beds in this area", f"{int(r.get('n_hospitals', 0))} · "
                                                    f"{r.get('beds_public', 0) + r.get('beds_private', 0):,.0f}"),
             ("Influenza per 100,000 (not scored)", f"{r.get('rate_influenza', np.nan):,.1f}"),
             ("Leptospirosis per 100,000 (not scored)", f"{r.get('rate_leptospirosis', np.nan):,.2f}"),
             ("Disease data level", {"city": "City's own DOH rate", "province": "Province's DOH rate"}.get(
                 r.get("disease_level"), "Regional median (no DOH data)")),
             ("Mean travel time to nearest hospital", f"{r.get('mean_minutes_any', np.nan):.0f} min "
              + ("(by road)" if float(r.get("road_share", 0) or 0) >= 0.5 else "(straight-line estimate)")),
             ("Bed shortfall vs DOH 1 bed per 800", f"{r['bed_gap_shortfall']:,.0f} beds")]
    if r.get("poverty_imputed", 0):
        rows.append(("Note", "Poverty imputed (city not in PSA SAE file)"))
    show_table(pd.DataFrame(rows, columns=["Indicator", "Value"]), hide_index=True, use_container_width=True)
with k2:
    st.markdown("**What drives the scores** (0–100)")
    dcols = [c for c in DOMAIN_LABELS if c in res.columns]
    dd = pd.DataFrame({"factor": [DOMAIN_LABELS[c] for c in dcols], "score": [float(r[c]) for c in dcols],
                       "kind": ["Need (higher = worse)" if c.startswith("hvs_") else "Capacity (higher = better)"
                                for c in dcols]})
    st.altair_chart(alt.Chart(dd).mark_bar(cornerRadiusEnd=4, size=16).encode(
        x=alt.X("score:Q", title=None, scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("factor:N", title=None, sort=list(dd["factor"]), axis=alt.Axis(labelLimit=260)),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Need (higher = worse)", "Capacity (higher = better)"],
                                                               range=["#d6604d", BLUE]),
                        legend=alt.Legend(orient="bottom", labelLimit=0)),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("score:Q", format=".0f", title="Score (0–100)")]).properties(height=330), use_container_width=True)
p1, p2, p3 = st.columns(3)


def pie(data, colors):
    """Donut chart: percentages inside the ring, legend in two columns below (nothing is clipped)."""
    keep = data["value"] > 0
    data, colors = data[keep], [c for c, k in zip(colors, keep) if k]
    base = alt.Chart(data).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("part:N", title=None, sort=list(data["part"]),
                        scale=alt.Scale(domain=list(data["part"]), range=colors),
                        legend=alt.Legend(orient="bottom", columns=2, labelLimit=0, symbolType="circle")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("part:N", title="Group"), alt.Tooltip("value:Q", format=",.0f", title="Value"),
                 alt.Tooltip("pct:Q", format=".1%", title="Share")])
    arcs = base.mark_arc(innerRadius=58, outerRadius=104, cornerRadius=3, stroke="white", strokeWidth=1.5)
    labels = base.mark_text(radius=81, size=12, color="white", fontWeight="bold").encode(
        text=alt.Text("pct:Q", format=".0%")).transform_filter("datum.pct >= 0.07")
    return (arcs + labels).properties(height=330, padding={"left": 6, "right": 6, "top": 6, "bottom": 6})


if "pop_children" in r.index and pd.notna(r.get("pop_children")):
    age = pd.DataFrame({"part": ["Children 0–14", "Youth 15–24", "Young adults 25–44", "Middle age 45–59", "Elderly 60+"],
                        "value": [r["pop_children"], r["pop_youth"], r["pop_young_adults"], r["pop_middle_age"],
                                  r["pop_elderly"]]})
    age_colors = ["#CE3550", "#F5C400", "#4F7FBE", "#9AA6B8", NAVY]
else:
    age = pd.DataFrame({"part": ["Children 0–4", "Ages 5–59", "Older adults 60+"],
                        "value": [r["pop_0_4"], r["population"] - r["pop_0_4"] - r["pop_60_plus"], r["pop_60_plus"]]})
    age_colors = ["#CE3550", "#B8C4DA", NAVY]
age["pct"] = age["value"] / age["value"].sum()
age["order"] = range(len(age))
p1.markdown("**Who lives here**")
p1.altair_chart(pie(age, age_colors), use_container_width=True)
beds = pd.DataFrame({"part": ["Public beds", "Private beds"],
                     "value": [r.get("beds_public", 0), r.get("beds_private", 0)]})
beds["pct"] = beds["value"] / max(beds["value"].sum(), 1)
beds["order"] = range(len(beds))
p2.markdown("**Licensed beds in this LGU**")
if beds["value"].sum() > 0:
    p2.altair_chart(pie(beds, [NAVY, "#F5C400"]), use_container_width=True)
else:
    p2.info("No licensed hospital or infirmary beds inside this city/municipality; residents rely on neighbouring LGUs.")
need_cols = [c for c in res.columns if c.startswith("hvs_") and not c.endswith("_pct")]
NEED_NAMES = {"hvs_demographic": "Age dependency", "hvs_socioeconomic": "Poverty", "hvs_geographic": "Isolation (GIDA)",
              "hvs_disease": "Disease burden"}
nd = pd.DataFrame({"part": [NEED_NAMES.get(c, c.replace("hvs_", "").title()) for c in need_cols],
                   "value": [float(r[c]) for c in need_cols]})
nd["pct"] = nd["value"] / max(nd["value"].sum(), 1e-9)
nd["order"] = range(len(nd))
p3.markdown("**What makes up its health need**")
p3.altair_chart(pie(nd, ["#CE3550", "#E8913A", BLUE, "#7B2FBE"]), use_container_width=True)

with st.expander("Compare with other cities and municipalities"):
    others = st.multiselect("Add up to 4 places", [x for x in labels if x != pick], max_selections=4,
                            help="Pick other cities or municipalities to see their scores side by side with this one.")
    cmp_ = res_lbl[res_lbl["label"].isin([pick] + others)].set_index("label")
    show = [c for c in ["gap_index", "HVS", "HCS", "beds_per_1000_all", "cov_any_30", "poverty_incidence",
                        "gida_share", "rate_tb", "rank_p05", "rank_p95"] if c in cmp_.columns]
    show_table(cmp_[show].rename(columns={"HVS": "HNI", "HCS": "HCI", "gap_index": "HGI"}).round(2),
                 use_container_width=True)
footer(meta)
