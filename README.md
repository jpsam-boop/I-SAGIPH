# I-SAGIPH

**Interactive Software for Analyzing Geographic Inadequacy in Philippine Healthcare**

I-SAGIPH combines public health, population, facility, and road data to map where healthcare needs outpace reachable hospital care across all 1,634 Philippine cities and municipalities, ranking where action is most urgent and explaining what drives each gap. It also lets planners test new facility sites and hospital closures, analyze equity, and helps residents find care and assistance near them.

> The Python package and conda environment are named `isagiph`, so every command below uses that name.

---

## Contents

1. [Why this matters](#why-this-matters)
2. [What I-SAGIPH does](#what-i-sagiph-does)
3. [Requirements](#requirements)
4. [Download and install](#download-and-install)
5. [Try it in 5 minutes (demo data)](#try-it-in-5-minutes-demo-data)
6. [Run it on the real data](#run-it-on-the-real-data)
7. [Using the app](#using-the-app)
8. [Outputs](#outputs)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Testing and reproducibility](#testing-and-reproducibility)
12. [Methods in brief](#methods-in-brief)
13. [Data and ethics](#data-and-ethics)
14. [Repository layout](#repository-layout)
15. [Limitations](#limitations)
16. [References and licence](#references-and-licence)

---

## Why this matters

The Universal Health Care Act (R.A. 11223) promises every Filipino access to care, but where care exists depends on decades of uneven investment. The Department of Health's planning target is one hospital bed per 800 people, and the Philippine Health Facility Development Plan 2020–2040 reports that about half of Filipinos are not within 30 minutes of a primary care facility (Department of Health, 2020). The data needed to see *where* need outruns care already exist, but in separate tables: facilities and beds from the DOH, population and poverty from the PSA, disease from FHSIS, and roads from OpenStreetMap. I-SAGIPH joins them, city by city, to show where to act first.

## What I-SAGIPH does

For all 1,634 cities and municipalities, I-SAGIPH builds three indices:

| Index | What it measures | Built from |
|---|---|---|
| **Health Need Index (HNI)** | How much care people are likely to need | Age structure, poverty, geographic isolation (GIDA), tuberculosis, pneumonia and hypertension rates |
| **Hospital Capacity Index (HCI)** | How much hospital care people can realistically reach | Beds within travel time (E2SFCA), 30-minute coverage, licensed primary care, Malasakit Center reach |
| **Healthcare Gap Index (HGI)** | Where need ranks above capacity | Need percentile minus capacity percentile, from −100 to +100 |

On top of the indices it:

- tests whether gaps cluster geographically (Moran's I, LISA, Getis-Ord Gi* with false-discovery-rate control);
- checks every result against 1,000 alternative modelling choices (Monte Carlo robustness);
- measures whether beds are fairly shared between poorer and richer areas (concentration index with a bootstrap confidence interval);
- compares each area's capacity with what similar places have, using machine-learning models tested on regions they never saw;
- recommends where new hospitals would help the most people (need-weighted maximal-covering optimisation);
- writes a findings report and a one-page policy brief, and serves everything through an interactive web app.

---

## Requirements

| Item | Needed |
|---|---|
| Operating system | Windows 10/11, macOS, or Linux |
| Python | 3.11 (installed automatically by the conda environment) |
| Package manager | [Anaconda](https://www.anaconda.com/download) or Miniconda (recommended), or `pip` |
| Disk space | About 5 GB free for the source data and outputs |
| Memory | 8 GB RAM minimum; 16 GB recommended for road-network routing |
| Internet | Only to download the project and the source data; the analysis itself runs offline |
| Poppler (`pdftotext`) | Reads the DOH and PCSO PDF tables. Installed by the conda environment; with `pip`, install it separately (see Troubleshooting) |

---

## Download and install

### 1. Get the project

**Option A: download the ZIP.** Download the project ZIP, then extract it, for example into your Downloads folder, so you have a folder named `isagiph`.

On Windows (Anaconda PowerShell Prompt):

```powershell
cd "$HOME\Downloads"
Expand-Archive .\isagiph.zip -DestinationPath . -Force
cd .\isagiph
```

**Option B: clone with Git.**

```bash
git clone <repo-url>
cd isagiph
```

### 2. Create the environment

Open **Anaconda Prompt** (Windows) or a terminal (macOS/Linux) in the `isagiph` folder:

```bash
conda env create -f environment.yml
conda activate isagiph
pip install -e .
```

The first command takes a few minutes and only needs to be run once. Every time you open a new terminal afterwards, run `conda activate isagiph` before any other command.

**Without conda:**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If a `requirements-lock.txt` file is present, `pip install -r requirements-lock.txt` installs the exact versions the results were produced with.

---

## Try it in 5 minutes (demo data)

The demo runs the whole pipeline on a **synthetic** dataset with the same structure as the real data. No downloads are needed. Demo numbers are not findings, and the app and reports label them clearly.

```bash
conda activate isagiph
python -m isagiph.pipeline --demo       # about 1 minute
python -m streamlit run app/app.py      # opens http://localhost:8501
```

---

## Run it on the real data

### Step 1. Download the source data

Put every file below in `data/manual/`. File names may differ slightly: the build step finds each file by a pattern, shown in the last column. Full citations with download links are in `docs/references.md`.

| Data set | Publisher | Where to get it | File pattern |
|---|---|---|---|
| Administrative boundaries table (COD-AB) | OCHA / PSA / NAMRIA | HDX: *Philippines – Subnational Administrative Boundaries* | `phl_admin_boundaries*.xlsx` |
| Level-3 boundary outlines (for the map) | OCHA / PSA / NAMRIA | Same HDX page (GeoJSON, shapefile, GeoPackage or geodatabase) | `phl_admin3.geojson` or similar |
| 2020 Census: household population by age group | PSA | psa.gov.ph | `4_Household*2020*.xlsx` |
| Poverty small area estimates 2023 | PSA | psa.gov.ph | `2_2023_SAE*.xlsx` |
| Licensed hospitals, infirmaries, primary care facilities, ambulance providers | DOH–HFSRB | hfsrb.doh.gov.ph | `Hospital_list_as_of*.xlsx`, `List_of_Infirmary*.xlsx`, `List_of_Primary_Care_Facilities*.xlsx`, `Free-standing_Ambulance*.xlsx` |
| FHSIS morbidity tables 2022 | DOH | DOH FHSIS annual report | `Morbidity*.pdf` |
| GIDA list 2023 | DOH | DOH | `*GIDA*.pdf` |
| Malasakit Centers list | PCSO | pcso.gov.ph | `PCSO_Malasakit*.pdf` |
| Health facilities (OpenStreetMap) | HOT | HDX: *hotosm_phl_health_facilities* | `health_facilities.gpkg` |
| Roads (OpenStreetMap) | HOT | HDX: *hotosm_phl_roads* | `roads.gpkg` (any `.gpkg` with "road" in its name) |

To keep the source files somewhere else, set `paths.manual` in `config/config.yaml` to that folder (use forward slashes, e.g. `C:/Users/you/Downloads/manual`).

### Step 2. Build, analyse, and open the app

Run these in order from the `isagiph` folder, with the environment active:

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python -m isagiph.build_real` | Links and cleans all sources, geocodes facilities, builds map outlines; writes `data/interim/` and `reports/data_quality.md` | ~5 min |
| 2 | `python -m isagiph.roadtimes --all` | Road-network travel times for every region from `roads.gpkg`; skips regions already done | hours (once) |
| 3 | `python -m isagiph.pipeline` | Indices, spatial statistics, robustness, machine learning, reports | ~2 min |
| 4 | `python -m streamlit run app/app.py` | Opens the app at http://localhost:8501 | seconds |

Step 2 is optional: without it, travel times use a documented straight-line estimate. With it, the pipeline's log reports how many barangays use road times.

**All in one (Windows, Anaconda PowerShell Prompt):** copy the whole block. It stops at the first failed step.

```powershell
& {
    cd "$HOME\Downloads\isagiph"
    conda activate isagiph
    pip install -e . --quiet
    foreach ($s in @("python -m isagiph.build_real", "python -m isagiph.roadtimes --all", "python -m isagiph.pipeline")) {
        Write-Host "`n=== $s ===" -ForegroundColor Cyan
        Invoke-Expression $s
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: $s" -ForegroundColor Red; return }
    }
    python -m streamlit run app/app.py
}
```

**With `make`** (macOS/Linux, or Windows with make installed): `make reproduce` runs steps 1–3, then `make app`.

**Next time**, to open the app without rebuilding:

```bash
conda activate isagiph
python -m streamlit run app/app.py
```

Re-run steps 1–3 only when the source data or `config/config.yaml` change.

---

## Using the app

Open http://localhost:8501 after starting the app. The menu at the top groups the pages:

| Menu | Page | Use it to |
|---|---|---|
| | **Home** | See the headline numbers and jump to every tool |
| **Healthcare Access** | Healthcare Access | Find nearby facilities, learn where to ask for medical assistance, prepare for a hospital visit, and read or share community reports |
| **Explore** | Map | Visualize where healthcare needs are highest and how hospital capacity is distributed, revealing mismatches between need and available care |
| | LGU Profile | Examine any city or municipality in depth and download a one-page PDF profile |
| | Priority List | Rank areas by the size of their gap, filter by region or province, and download the list as CSV |
| | ISAGIP Simulator | Place hypothetical hospitals or health units, or let the optimiser suggest the best sites, and see how access and the gap change |
| **Analyze** | Regression Explorer | Compare each area's actual capacity with what similar places have, using models tested on regions they never saw |
| | Health Relationships | Explore how poverty, isolation, age and disease relate to access, and test whether beds are fairly shared (Equity tab) |
| | Crisis Impact | Simulate a hospital closure or bed reduction and see which communities would lose access |
| **Learn** | How to Read | What each index, colour and score means |
| | How It Works | Data sources, methods, assumptions and limitations |
| | About | Purpose, audience and data |

### Typical workflows

**Find where to act first.** Open **Priority List**, filter to your region, and sort by gap. Open the top areas in **LGU Profile** to see what drives each gap, then download the PDF profile to share with local officials.

**Read the map.** On **Map**, choose an index, pick a region or province, and press **Apply**. Click any area to open its profile in the side panel. Hover over underlined labels for plain-language definitions. Choose **Bed need met** to see how far each area is from enough beds, not just how it ranks.

**Test a new hospital.** On **ISAGIP Simulator**, choose a facility type, then click the map or pick a place below it. To let the model choose, set the number of hospitals and press **Find best sites**, then **Add these sites to the map**. The impact numbers below the map come from the full model.

**Check fairness.** On **Health Relationships**, open the Equity tab to compare access across poverty groups, isolated areas, age structure and regions.

**Plan for disruptions.** On **Crisis Impact**, choose a hospital and a scenario (closure or bed reduction) to see the affected population and how the gap changes.

**Help residents.** Share **Healthcare Access**: residents can search for facilities by place and service, find official assistance links, use the visit checklist, and post anonymous reports on access barriers. For demonstrations, `python -m isagiph.community` adds labelled sample posts; `python -m isagiph.community --remove-samples` deletes them.

### Tips

- After rebuilding the data, refresh the browser with **Ctrl+F5** so the app reloads its cached files.
- Indices are **rankings** among areas: a low gap or high capacity score means better than other areas, not enough care. Use **Bed need met** and the bed targets to judge adequacy.
- To stop the app, press **Ctrl+C** in the terminal.

---

## Outputs

| File | Contents |
|---|---|
| `reports/findings.md` | Headline numbers, clustering tests, regional table, priority areas, equity (with confidence interval), bottlenecks and actions |
| `reports/policy_brief.md` | One page for health offices: the top priority areas per region, what holds each back, and what to do |
| `reports/data_quality.md` | Every match rate, imputation and unlinked record from the build |
| `reports/road_vs_straight_*.md` | Per region: how road travel times compare with straight-line estimates |
| `data/processed/lgu_results.csv` | All indices and indicators for every city and municipality |
| `data/processed/ml_*.csv`, `ml_metrics.json` | Machine-learning predictions, importance and metrics |
| `data/processed/run_meta.json` | Settings and timestamp of the run |

---

## Configuration

Every analytical assumption is in `config/config.yaml`, with comments. The ones most often changed:

| Setting | Meaning |
|---|---|
| `paths.manual` | Folder with the source files (default `data/manual`) |
| `roads.file`, `roads.layer` | Which roads GeoPackage and layer to route on, if several exist |
| `roads.speeds_kmh` | Travel speed per road class (km/h) |
| `accessibility` | Catchment times (30 min for Level 1, 60 min for Level 2–3) and the fallback speed for straight-line estimates |
| `benchmarks.beds_per_1000` | DOH minimum (1.25, i.e. one bed per 800 people) |
| `boundaries.simplify_deg` | Map-outline detail; larger = lighter map |
| `sensitivity.n_runs` | Number of Monte Carlo robustness runs (default 1,000) |
| `seed` | Random seed; fixed so results repeat exactly |

After changing settings, re-run `python -m isagiph.pipeline` (and `build_real` if paths changed).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognized | Use `python -m streamlit run app/app.py`, and make sure `conda activate isagiph` was run first |
| `File does not exist: app/app.py` | You are in the wrong folder. Run `cd` into the `isagiph` folder that contains `app/` |
| `FileNotFoundError: ... data/interim/lgus.csv` | Run `python -m isagiph.build_real` before the pipeline or road times |
| `No file matching ... in data/manual` | A source file is missing or named very differently; check Step 1, or set `paths.manual` |
| `No roads GeoPackage found` | Put `roads.gpkg` in `data/manual/`, or set `roads.file` in `config/config.yaml` |
| Pipeline says *straight-line* instead of *road network* | Road-time files are missing from `data/interim/`; run `python -m isagiph.roadtimes --all`, then the pipeline |
| `pdftotext` not found | Install Poppler: `conda install -c conda-forge poppler` |
| Map shows hexagons, not city outlines | Put the level-3 boundary file (e.g. `phl_admin3.geojson`) in `data/manual/` and re-run `build_real` |
| App shows old results | Press **Ctrl+F5** in the browser, or stop the app and start it again |
| `Port 8501 is already in use` | Close the other app window, or run `python -m streamlit run app/app.py --server.port 8502` |
| Out of memory during road times | Close other programs and re-run `roadtimes --all`; finished regions are skipped |
| Many spreadsheet warnings during the build | Harmless (phone numbers stored as dates in the DOH files); they are silenced in the current version |

---

## Testing and reproducibility

```bash
pytest -q                 # unit and smoke tests (also run automatically on GitHub Actions)
make lock                 # record exact package versions in requirements-lock.txt
```

- All analytical parameters are in one file (`config/config.yaml`), and the random seed is fixed, so the same inputs give the same results.
- `make reproduce` rebuilds every result from the source files in the correct order.
- `reports/data_quality.md` documents every data-linking decision for each build.

---

## Methods in brief

| Stage | Method | Reference |
|---|---|---|
| Access | Enhanced two-step floating catchment area (E2SFCA) with road-network travel times | Luo & Qi (2009) |
| Need and capacity indices | Normalisation, PCA with varimax weighting, geometric-mean aggregation | OECD & JRC (2008); Nicoletti et al. (2000); Kaiser (1958) |
| Gap | Need percentile minus capacity percentile | Wang & Luo (2005) |
| Spatial patterns | Moran's I, LISA, Getis-Ord Gi*, Benjamini–Hochberg FDR | Moran (1950); Anselin (1995); Getis & Ord (1992) |
| Robustness | 1,000-run Monte Carlo over weights, scaling and catchments | Saisana et al. (2005) |
| Equity | Concentration index with bootstrap confidence interval | Wagstaff et al. (1991); O'Donnell et al. (2008) |
| Machine learning | Linear, ridge, random forest, gradient boosting; nested spatial cross-validation | Breiman (2001); Roberts et al. (2017) |
| Missing disease data | Label propagation on a neighbour graph, validated by hiding provinces | Zhou et al. (2004) |
| Facility siting | Need-weighted maximal covering location problem, greedy with a (1 − 1/e) guarantee | Church & ReVelle (1974); Nemhauser et al. (1978) |

Full details: `docs/statistical_models.md` and `docs/methodology.md`.

---

## Data and ethics

- Only public, aggregate data are used. There are no patient or personal records.
- Source files are not redistributed in this repository (`data/` is excluded by `.gitignore`); `docs/references.md` lists where each can be obtained, and `docs/data_card.md` lists each source with its reference period, use and licence.
- Community reports in the app are anonymous, unverified, and stored only on the machine running it. Uploaded photos are resized and stripped of location (GPS) data.

---

## Repository layout

| Path | Purpose |
|---|---|
| `config/config.yaml` | Every analytical parameter, with comments |
| `src/isagiph/build_real.py`, `schema.py`, `boundaries.py` | Source files to validated tables; map outlines |
| `src/isagiph/roadtimes.py`, `accessibility.py` | Road-network routing, travel times, E2SFCA, coverage |
| `src/isagiph/indices.py`, `impute.py` | Need, capacity and gap indices; missing disease rates |
| `src/isagiph/spatial_stats.py`, `sensitivity.py`, `validation.py` | Spatial statistics, robustness, outcome validation (ready for outcome data) |
| `src/isagiph/ml.py`, `equity.py`, `optimize.py`, `simulate.py` | Machine learning, equity statistics, site optimisation, what-if simulator |
| `src/isagiph/report.py`, `pipeline.py` | Reports and orchestration |
| `src/isagiph/community.py` | Community reports and announcements |
| `app/` | Streamlit app: `app.py` (menu), `common.py` (shared code), `views/` (one file per page), `assets/` (logo) |
| `tests/` | Unit and smoke tests (run in CI) |
| `docs/` | Code guide, statistical models, methodology, data card, references, demo video script |
| `reports/` | Generated reports |
| `Makefile`, `environment.yml`, `requirements.txt` | Build commands and dependencies |

---

## Limitations

- **Travel times are modelled** from road-class speeds on OpenStreetMap roads, with a stated fallback speed where no route exists. They do not capture schedules, fares, waiting, or ferries.
- **Licensed beds are not staffed or available beds**, and the facility list is not a live inventory.
- **No outcome data are integrated yet**, so the outcome-validation step reports no result. Adding a province-level outcome (for example, facility-based delivery from the NDHS) would let the index be tested against health outcomes.
- **Disease rates are reported by province or city**, so municipalities share their province's rate, and reported rates partly reflect how completely each area reports cases.
- **Indices are rankings**, not measures of adequacy: an area can rank well and still lack enough care.
- **City-level averages hide inequality inside each city or municipality.**
- Results show where to look first; they should prompt verification with provincial and city health offices, not replace it.

---

## References and licence

Full references in APA 7th edition (literature, legislation and data sets) are in `docs/references.md`.

Code is released under the MIT License (see `LICENSE`). Data remain under their publishers' terms; OpenStreetMap data © OpenStreetMap contributors, available under the Open Database License.
