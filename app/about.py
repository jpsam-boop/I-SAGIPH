"""About: purpose, audience, and data."""
import streamlit as st

from common import APP_NAME, ASSETS, BRAND_HTML, FULL_NAME, footer, page_header, require_data

res, fac, meta, valid, weights, gj, pts = require_data()
page_header("About", f"About {BRAND_HTML}", f"<b>{BRAND_HTML}</b>: {FULL_NAME}.")

st.image(str(ASSETS / "logo_full.png"), width=150)
st.markdown(f"""
<div class="about">
  <div>
    <h3>Why {APP_NAME}</h3>
    <p>The Universal Health Care Act (R.A. 11223) and the Malasakit Centers Act (R.A. 11463) aim to bring care to
       every Filipino, but facility, population, disease, and socioeconomic data are published separately. {APP_NAME} puts them
       on one map so planners can see where need and capacity diverge.</p>
    <h3>Who it is for</h3>
    <p>LGU and provincial health offices, DOH regional offices, Malasakit Center administrators, researchers, and citizens.
       Findings show where to look, not what to decide.</p>
  </div>
  <div>
    <h3>Built on open data</h3>
    <p>Every number comes from public sources: the PSA, the DOH, PCSO, OCHA's administrative boundaries, and
       OpenStreetMap. The code and methods are documented so the analysis can be checked and repeated.</p>
  </div>
</div>""", unsafe_allow_html=True)
footer(meta)
