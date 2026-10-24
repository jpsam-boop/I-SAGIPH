"""Concise guide to using I-SAGIPH and reading its indices, maps, and analytical outputs."""
import streamlit as st

from common import footer, page_header, require_data

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("Learn · Quick guide", "How to use I-SAGIPH",
            "Start with a question, choose the matching section, then read the result alongside its assumptions and data limits.")

st.markdown("### 1. Choose the right section")
explore, analyze, community = st.columns(3)
with explore:
    st.markdown("**Explore**")
    st.write("See the map, open a city or municipality profile, compare priority areas, or test a hypothetical facility investment in the simulator.")
    st.page_link("views/map_page.py", label="Open Explore", icon=":material/map:", use_container_width=True)
with analyze:
    st.markdown("**Analyze**")
    st.write("Check regression predictions, explore associations between access and health indicators, or test the effects of a facility closure or capacity loss.")
    st.page_link("views/regression_explorer.py", label="Open Analyze", icon=":material/query_stats:", use_container_width=True)
with community:
    st.markdown("**Community**")
    st.write("Find a facility, review medical-assistance links, prepare for a visit, and view or share community information.")
    st.markdown('<a href="citizen?step=community" target="_self">Open Healthcare Access →</a>', unsafe_allow_html=True)

st.markdown("### 2. Read the three main indices")
idx1, idx2, idx3 = st.columns(3)
with idx1:
    st.markdown("**Health Need Index (HNI) · 0–100**")
    st.write("Higher means an area has a higher relative need for healthcare, based on factors such as age, poverty, geographic isolation, and selected disease indicators.")
with idx2:
    st.markdown("**Hospital Capacity Index (HCI) · 0–100**")
    st.write("Higher means more healthcare capacity is estimated to be reachable compared with other LGUs. It is not a guarantee that enough care is available.")
with idx3:
    st.markdown("**Healthcare Gap Index (HGI) · −100 to +100**")
    st.write("Need percentile minus capacity percentile. Positive values mean need ranks ahead of reachable capacity; larger positive values indicate a wider relative gap.")

st.markdown("### 3. Understand map colours and labels")
st.markdown("""
- **Red on the Health Need layer:** greater relative healthcare need.
- **Blue/teal on the Hospital Capacity layer:** more modeled capacity within reach than other areas.
- **Purple on the Healthcare Gap layer:** darker means need is further ahead of capacity relative to other LGUs. Pale purple means a smaller gap, not necessarily adequate care.
- **Bed need met:** an absolute-style comparison showing reachable beds as a share of the 2040 goal of 4 beds per 1,000 people. This is different from the relative indices.
- **Priority area:** an LGU classified as high need and low capacity under the configured need-capacity grid. **Stable** means the classification persists in at least 80% of the model's robustness runs; it does not mean the data are perfect.
""")

st.markdown("### 4. Use the analytical tools")
with st.container(border=True):
    st.markdown("**Regression Explorer**")
    st.write("Each dot is an LGU. The closer a dot is to the diagonal, the closer the model prediction is to the E2SFCA-derived reference value. MAE is the average absolute error; RMSE gives larger misses extra weight; R² summarises predictive agreement. These metrics evaluate reproduction of the derived accessibility measure, not actual patient outcomes.")
    st.markdown("**Health Relationships**")
    st.write("Choose an access measure and a health indicator. Correlation describes how values move together; adjusted estimates account for selected measured factors. Neither one alone proves that access caused the health outcome.")
    st.markdown("**Crisis Impact**")
    st.write("Remove a facility or reduce its licensed beds to estimate how modeled travel-time access may change. Population counts describe residents affected by the chosen threshold in the model—not confirmed patients who would miss care.")
    st.markdown("**ISAGIP Simulator**")
    st.write("Place a hypothetical hospital or primary-care facility and compare before/after results. This tests an assumption; it is not a construction decision or a costed feasibility study.")

st.markdown("### 5. Check the evidence before acting")
st.info("Travel times are modeled using the road network where a usable route is available and configured transport assumptions, with a fallback where routing is unavailable. The estimate of residents beyond 30 minutes from a hospital is compared with the supplied external DOH reference figure. Confirm data dates, route coverage, staffing, services, and current facility status before using a result for planning.")
st.caption("A map or index points to places that may need further investigation. It does not measure every resident's journey, guarantee facility availability, or decide where public money should go.")
footer(meta)
