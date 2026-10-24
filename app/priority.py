"""Priority list: LGUs ranked by the Healthcare Gap Index, with filters and CSV export."""
import streamlit as st

from common import GLOSSARY, GOAL_2040, cascade, footer, page_header, require_data, show_table, term

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("Priority list", "Where to look first")

if "prio_applied" not in st.session_state:
    st.session_state.prio_applied = {"regions": [], "provinces": [], "only_priority": True, "only_stable": False}
fragment = getattr(st, "fragment", lambda f: f)


@fragment
def prio_controls() -> None:
    with st.container(border=True):
        regions, provinces = cascade(st.columns(2), sorted(res["region"].unique()), res, "prio")
        g1, g2, g3 = st.columns([1, 1, 1])
        only_priority = g1.toggle("Priority areas only", value=st.session_state.prio_applied["only_priority"],
                                  help=GLOSSARY["priority"])
        only_stable = g2.toggle("Stable results only", value=st.session_state.prio_applied["only_stable"],
                                help=GLOSSARY["stable"])
        if g3.button("Apply", type="primary", use_container_width=True):
            st.session_state.prio_applied = {"regions": regions, "provinces": provinces,
                                             "only_priority": only_priority, "only_stable": only_stable}
            st.rerun()


prio_controls()
_a = st.session_state.prio_applied
regions, provinces, only_priority, only_stable = _a["regions"], _a["provinces"], _a["only_priority"], _a["only_stable"]

view = res.copy()
if regions:
    view = view[view["region"].isin(regions)]
if provinces:
    view = view[view["province"].isin(provinces)]
if only_priority:
    view = view[view["priority"]]
if only_stable:
    view = view[~view["unstable"]]

view = view.assign(pct_goal=(100 * view["beds_per_1000_all"] / GOAL_2040).round(0),
                   stable=~view["unstable"], cov_any_30=(100 * view["cov_any_30"]).round(0),
                   cov_malasakit_60=(100 * view["cov_malasakit_60"]).round(0),
                   priority_freq=(100 * view["priority_freq"]).round(0))
cols = ["name", "province", "region", "population", "gap_index", "HVS", "HCS", "pct_goal", "bed_gap_shortfall",
        "cov_any_30", "cov_malasakit_60", "priority_freq", "rank_p05", "rank_p95", "stable"]
cols = [c for c in cols if c in view.columns]
table = view.sort_values("gap_index", ascending=False)[cols].rename(columns={
    "pct_goal": "% of 2040 bed goal met", "bed_gap_shortfall": "Bed shortfall (DOH minimum)",
    "cov_any_30": "% within 30 min of a hospital", "cov_malasakit_60": "% within 60 min of a Malasakit Center",
    "priority_freq": "Priority in % of model runs", "rank_p05": "Best rank", "rank_p95": "Worst rank"}).round(1)
st.caption(f"{len(table):,} areas · {view['population'].sum() / 1e6:,.1f}M people")
show_table(table, use_container_width=True, height=560, hide_index=True)
st.download_button("⬇ Download CSV", table.to_csv(index=False), "i-sagiph_priority_list.csv", "text/csv")
footer(meta)
