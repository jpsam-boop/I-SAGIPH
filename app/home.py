"""Home: animated project headline, horizontally scrollable headline statistics, and three entry points."""
import html
import random
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from common import (BLUE, BRAND_HTML, NAVY, RED, YELLOW, HERO_SVG, footer, headline_numbers,
                    require_data, section)

res, fac, meta, valid, weights, gj, pts = require_data()
n = headline_numbers(res)
population = float(res["population"].sum()) if "population" in res else 0.0
within30 = max(0.0, 100.0 - float(n["beyond30"]))
beds_within = (1000 * float(res["accessible_beds"].sum()) / population) if population and "accessible_beds" in res else float("nan")
short_2040 = float(pd.to_numeric(res.get("bed_gap_2040", pd.Series(0, index=res.index)), errors="coerce").fillna(0).sum())

# Pick one of two safe, diagonal edge layouts per Streamlit session. The centre of the
# map illustration stays clear, and CSS alternates between exactly two statistic chips.
layout = st.session_state.setdefault("_hero_chip_layout", random.SystemRandom().choice((0, 1)))
chip_layouts = [
    ("left:-14px;bottom:18%;top:auto;right:auto;", "right:-10px;top:10%;bottom:auto;left:auto;",
     "left:-12px;top:10%;bottom:auto;right:auto;", "right:-8px;bottom:14%;top:auto;left:auto;"),
    ("left:-12px;top:10%;bottom:auto;right:auto;", "right:-8px;bottom:14%;top:auto;left:auto;",
     "left:-14px;bottom:18%;top:auto;right:auto;", "right:-10px;top:10%;bottom:auto;left:auto;"),
    ("left:0;top:2%;bottom:auto;right:auto;", "right:0;top:22%;bottom:auto;left:auto;",
     "left:0;bottom:8%;top:auto;right:auto;", "right:0;bottom:2%;top:auto;left:auto;"),
    ("left:0;bottom:26%;top:auto;right:auto;", "right:0;top:3%;bottom:auto;left:auto;",
     "left:0;top:18%;bottom:auto;right:auto;", "right:0;bottom:24%;top:auto;left:auto;"),
]
chip_styles = chip_layouts[int(layout)]

# Animated title and map statistics. CSS swaps two chips at a time and keeps them at the art's outer edges.
st.markdown(f"""
<div class="hero">
  <div>
    <span class="kicker">Healthcare gap mapping · Philippines</span>
    <h1 style="margin-top:14px">Where does care fall short of need?</h1>
    <div class="acro"><b>{BRAND_HTML}</b>: Interactive Software for Analyzing Geographic Inadequacy in Philippine Healthcare</div>
    <p class="lead">A clearer picture of healthcare access across Philippine cities and municipalities—using population, poverty, isolation, facility capacity, and transport assumptions to show where needs and reachable care diverge.</p>
    <div class="cta"><a class="btn solid" href="map" target="_self">Explore the map</a>
      <a class="btn line" href="how-to-read" target="_self">How to use I-SAGIPH</a></div>
    <div class="credits">Built from public datasets and documented transport assumptions; results are planning estimates, not live service availability.</div>
  </div>
  <div class="art">{HERO_SVG}
    <div class="chip c1 pair-a" style="{chip_styles[0]}"><b>{n['stable']:,}</b><span class="dot" style="background:{RED}"></span>areas consistently flagged as priorities</div>
    <div class="chip c2 pair-a" style="{chip_styles[1]}"><b>{n['beyond30']:.1f}%</b><span class="dot" style="background:{BLUE}"></span>of people beyond 30 min from a hospital</div>
    <div class="chip c3 pair-b" style="{chip_styles[2]}"><b>{n['beds_short']:,.0f}</b><span class="dot" style="background:{YELLOW}"></span>estimated beds below the 1.25-per-1,000 benchmark</div>
    <div class="chip c4 pair-b" style="{chip_styles[3]}"><b>{n['pri_pop_m']:,.1f}M</b><span class="dot" style="background:{RED}"></span>people in priority areas</div>
  </div>
</div>""", unsafe_allow_html=True)

section("At a glance", "Key numbers behind the map", "Swipe horizontally to explore the headline indicators. The first visible card highlights the current focus.")
stat_cards = [
    (f"{n['pri_pop_m']:,.1f}M", "People in priority areas", "LGUs with high relative need and low relative capacity"),
    (f"{n['stable']:,}", "Stable priority areas", "Flagged as priorities in at least 80% of sensitivity runs"),
    (f"{n['beyond30_m']:,.1f}M", "Beyond 30 minutes", f"Estimated {n['beyond30']:.1f}% of the population beyond a hospital catchment"),
    (f"{within30:.1f}%", "Within 30 minutes", "Estimated population share within the modeled hospital travel threshold"),
    (f"{n['beds_short']:,.0f}", "Beds below benchmark", "Estimated shortfall against 1.25 hospital beds per 1,000 people"),
    (f"{short_2040/1000:,.0f}k", "Beds below 2040 goal", "Estimated shortfall against 4 beds per 1,000 people"),
    (f"{len(res):,}", "Cities and municipalities", "LGUs included in this processed run"),
    (f"{beds_within:.2f}" if beds_within == beds_within else "N/A", "Reachable beds per 1,000", "Population-weighted national estimate from modeled access"),
]
card_html = "".join(
    f"<article class='stat-rail-card{' is-active' if i == 0 else ''}'><div class='value'>{html.escape(v)}</div>"
    f"<div class='label'>{html.escape(label)}</div><div class='note'>{html.escape(note)}</div></article>"
    for i, (v, label, note) in enumerate(stat_cards)
)
components.html(f"""
<div class="stat-rail" id="isagiph-stat-rail" aria-label="Scrollable healthcare statistics">{card_html}</div>
<style>
*{{box-sizing:border-box}} body{{margin:0;font-family:Inter,Arial,sans-serif;background:transparent}}
.stat-rail{{display:flex;gap:12px;overflow-x:auto;overflow-y:hidden;padding:4px 2px 12px;scroll-snap-type:x mandatory;scrollbar-width:thin;}}
.stat-rail-card{{flex:0 0 210px;min-height:126px;scroll-snap-align:start;border:1px solid #ECE5D6;border-radius:18px;padding:15px 16px;background:#fff;color:#002157;transition:background .18s ease,color .18s ease;}}
.stat-rail-card .value{{font-size:1.8rem;font-weight:900;letter-spacing:-.03em;line-height:1.05;}}
.stat-rail-card .label{{font-size:.82rem;font-weight:750;margin-top:9px;line-height:1.25;}}
.stat-rail-card .note{{font-size:.72rem;color:#5A6B85;margin-top:4px;line-height:1.35;}}
.stat-rail-card.is-active{{background:#002157;border-color:#002157;color:#fff;}}
.stat-rail-card.is-active .note{{color:#C8D4E9;}}
</style>
<script>
(function(){{const rail=document.getElementById('isagiph-stat-rail');if(!rail)return;const cards=Array.from(rail.querySelectorAll('.stat-rail-card'));
function updateActive(){{if(!cards.length)return;const left=rail.getBoundingClientRect().left;let best=cards[0],bestDist=Infinity;cards.forEach(c=>{{const d=Math.abs(c.getBoundingClientRect().left-left);if(d<bestDist){{best=c;bestDist=d;}}}});cards.forEach(c=>c.classList.toggle('is-active',c===best));}}
rail.addEventListener('scroll',()=>window.requestAnimationFrame(updateActive),{{passive:true}});window.addEventListener('resize',updateActive);updateActive();}})();
</script>""", height=164, scrolling=False)

section("The three indices", "How to read the scores", "These scores compare healthcare need and estimated reachable capacity across LGUs. They are planning indicators—not a guarantee that care is sufficient.")
st.markdown("""
<div class="cards three">
  <div class="idx-card" style="--c:#C92F43">
    <div class="t">01 · Who needs care?</div><div class="v">HNI</div>
    <p>Estimates relative healthcare need using factors such as young children and older adults, poverty, geographic isolation, and selected disease indicators.</p>
    <div class="bar"><div style="width:100%;background:linear-gradient(90deg,#FFE4C9,#F46F43,#C00000)"></div></div>
    <div class="scale-l"><span>0 · lower relative need</span><span>100 · higher need</span></div>
  </div>
  <div class="idx-card" style="--c:#2A8CBA">
    <div class="t">02 · What care is within reach?</div><div class="v">HCI</div>
    <p>Summarises reachable beds, travel-time coverage, primary-care availability, and listed Malasakit Center reach compared with other LGUs.</p>
    <div class="bar"><div style="width:100%;background:linear-gradient(90deg,#F2F6BB,#69C7BE,#23399B)"></div></div>
    <div class="scale-l"><span>0 · lower relative capacity</span><span>100 · higher capacity</span></div>
  </div>
  <div class="idx-card" style="--c:#6C2A9A">
    <div class="t">03 · Where does need exceed capacity?</div><div class="v">HGI</div>
    <p>Subtracts capacity percentile from need percentile. Positive values mean need ranks higher than reachable capacity; a smaller gap does not guarantee adequate care.</p>
    <div class="bar"><div style="width:100%;background:linear-gradient(90deg,#F1EAF7,#B28BD8,#42117B)"></div></div>
    <div class="scale-l"><span>−100 · capacity ranks higher</span><span>+100 · need ranks higher</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

section("Start here", "Choose what you want to do", "Three entry points. Use the links under each card to jump straight to the relevant tool.")
explore_col, analyze_col, community_col = st.columns(3, gap="medium")
with explore_col:
    st.markdown("""<div class="card nav-hub"><span class="no">01</span><h3>Explore</h3><span class="aka">See where care is within reach</span><p>Explore the national map, inspect a city or municipality, compare priority areas, or test a possible facility placement.</p></div>""", unsafe_allow_html=True)
with analyze_col:
    st.markdown("""<div class="card nav-hub"><span class="no">02</span><h3>Analyze</h3><span class="aka">Understand the evidence</span><p>Check how well the model reproduces its reference, explore associations with health indicators, and test healthcare disruption scenarios.</p></div>""", unsafe_allow_html=True)
with community_col:
    st.markdown("""<div class="card nav-hub"><span class="no">03</span><h3>Community</h3><span class="aka">Find help and share information</span><p>Search facilities, learn where to ask about medical assistance, prepare for a visit, and exchange community information.</p></div>""", unsafe_allow_html=True)

shortcut_cols = st.columns(4, gap="small")
for col, page, label, icon in zip(shortcut_cols,
        ["views/map_page.py", "views/profile.py", "views/priority.py", "views/simulator.py"],
        ["Explore map", "Look up an LGU", "Priority list", "Test placement"],
        [":material/map:", ":material/location_city:", ":material/format_list_numbered:", ":material/add_location_alt:"]):
    with col:
        st.page_link(page, label=label, icon=icon, use_container_width=True)

st.markdown("""<div class="analysis-note"><b>How to interpret the numbers:</b> the indices compare LGUs with one another; they do not certify that an area has enough healthcare. Travel times are modeled estimates, and a facility listing does not confirm current service availability.</div>""", unsafe_allow_html=True)
footer(meta)
