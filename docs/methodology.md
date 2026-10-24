# Methodology

## 1. Unit of analysis

The primary unit is the city/municipality (about 1,600 LGUs), joined across all sources by PSGC code. Barangays are used as demand points for accessibility, and results are aggregated to LGUs with population weights. Province-level sources (NDHS) are used only for validation.

## 2. Spatial accessibility (E2SFCA)

Following Luo and Qi (2009), with demand points *k* (barangay centroids, population *P*), hospitals *j* (licensed beds *S*), and travel time *d*:

- Step 1: R_j = S_j / Σ_k P_k · W(d_kj), over points within facility j's catchment
- Step 2: A_i = Σ_j R_j · W(d_ij)
- Decay: W(d) = [exp(−½(d/d₀)²) − exp(−½)] / [1 − exp(−½)] for d ≤ d₀, else 0

Catchments are d₀ = 30 min for Level 1 hospitals and 60 min for Level 2/3 hospitals. Total accessible beds always equal total supplied beds; `tests/test_accessibility.py` checks this conservation property.

**Travel times.** Preferred: use the Philippines roads GeoPackage with `python -m isagiph.roadtimes --all`. Road-class speed assumptions are set in `src/isagiph/roadtimes.py` / `config/config.yaml`; an explicit access/egress speed is used for the short legs between barangay/facility points and the nearest mapped road. Put the road line GeoPackage in `data/manual/`. If more than one plausible roads GeoPackage is present, set `roads.file` (and if necessary `roads.layer`) to the exact roads file/layer in `config/config.yaml`. The routing selector explicitly excludes `health_facilities.gpkg`, which is only used for health-facility coordinate matching.

Where usable route data are unavailable, fallback travel time is straight-line kilometres × configured circuity (default 1.4) ÷ configured generalized effective speed (default 15 km/h). This is an explicit assumption intended to represent mixed local travel patterns, not a direct mode-specific route: public-transport schedules, wait/transfer time, fares, and boat/ferry connections are not individually modeled. These are explicit transport assumptions. They are **built from documented road-network and mixed-local-transport assumptions**; DOH facility data define the facility inventory, and any coverage statistic is used only as an external context/check where the measure is comparable. The PHFDP's commonly cited 30-minute statistic concerns RHUs/health centres (primary care), rather than hospitals, so it is contextual only and an external reference for comparison with the project's estimated population beyond the 30-minute hospital threshold, when definitions align. Island LGUs without road connections must be interpreted cautiously because the fallback does not penalize water crossings.

## 3. Indicators

| Score | Domain | Indicator | Treatment |
|---|---|---|---|
| HVS | Demographic | % aged 0–4; % aged 60+ | Share of population |
| HVS | Socioeconomic | Poverty incidence | PSA small-area estimates where available |
| HVS | Disease | Tuberculosis, pneumonia, and hypertension per 100,000 (DOH FHSIS 2022) | Province or city rate; log(1+x) |
| HVS | Geographic | Share of barangays on the DOH 2023 GIDA list | 0–1 |
| HCS | Beds | E2SFCA beds per 1,000 (all; Level 2+) | log(1+x) |
| HCS | Coverage | % population ≤ 30 min to any hospital; ≤ 60 min to Level 2+ | Share |
| HCS | Primary care | RHU/health centres per 10,000 | log(1+x) |
| HCS | Social protection | % population ≤ 60 min to a Malasakit Center | Share |

Validation outcomes (e.g., NDHS facility-based delivery) are never used as inputs.

### Real-data implementation notes (`build_real.py`)

- **LGU frame**: COD-AB adm3 (1,642 cities/municipalities) with centroids. Sources are linked by PSGC where available, otherwise by normalised names within the region and province, with a fuzzy fallback restricted to the province.
- **Population**: 2020 CPH household population by age group. LGUs created after 2020 (the BARMM Special Geographic Area) have no census match and are excluded.
- **Demand points**: barangay centroids, with LGU population split equally across barangays. Barangay-level population would refine this.
- **Poverty**: PSA 2023 city/municipal estimates. Manila's 14 districts are averaged. Highly urbanised cities, which are absent from the file, are imputed with the regional median and flagged as `poverty_imputed`.
- **Beds**: DOH licensed hospitals (General class only; specialty hospitals excluded) plus licensed infirmaries, which count as first-level inpatient capacity.
- **Facility coordinates**: name-matched to HOT OSM hospitals and clinics in the same LGU; otherwise placed at the LGU centroid. `reports/data_quality.md` reports how many fall in each group.
- **Malasakit Centers**: PCSO list (November 2022), name-matched within region.
- **Disease rates**: DOH FHSIS 2022 morbidity tables, which report cases and rates per 100,000 for provinces and cities. Each LGU takes its city's own row if it has one, otherwise its province's. DOH marks missing reports as 0 ("No Data"), so zero totals are treated as missing and filled with the regional median (flagged per disease). Tuberculosis, pneumonia, and hypertension are scored; influenza and leptospirosis are shown but not scored, because most areas report "No Data" for them.
- **Not scored**: free-standing ambulances (134 providers only). The domain floor is 10, so one zero-inflated domain cannot dominate the geometric mean.

## 4. Composite construction: three indices

| Index | Plain name | Built from |
|---|---|---|
| **Health Need Index (HNI)** (column `HVS`) | Who needs care | Tuberculosis, pneumonia, and hypertension rates, % aged 0–4, % aged 60+, poverty, GIDA share |
| **Hospital Capacity Index (HCI)** (column `HCS`) | What care is within reach | E2SFCA beds per 1,000 (all; Level 2+), travel-time coverage, licensed primary care, Malasakit reach |
| **Healthcare Gap Index (HGI)** (column `gap_index`) | Gap | pct(HNI) − pct(HCI) |

This follows the integrated approach of Wang and Luo (2005): factor analysis of population characteristics gives a need score, a floating catchment model gives spatial access, and the two are combined to identify shortage areas.

**Consolidation model (weighting = `pca`, the default).** The procedure follows OECD and JRC (2008, Step 6) and Nicoletti et al. (2000), the same family as the SoVI (Cutter et al., 2003) and PCA-based equity indices (Vyas & Kumaranayake, 2006):

1. Normalise and direction-align the indicators (steps below).
2. Run PCA on the correlation matrix and keep factors with eigenvalue > 1 (Kaiser criterion).
3. Apply varimax rotation (Kaiser, 1958).
4. Assign each indicator to the factor on which it loads most. Its weight within that factor is its squared loading, scaled so the weights in each factor sum to 1.
5. Weight each factor by its share of the explained variance.
6. Combine the factors with a weighted geometric mean.

The weights from each run are saved to `data/processed/index_weights.csv` and shown in the dashboard. Equal weights (`weighting: equal`) serve as the alternative in the sensitivity analysis.

In a prior run, two factors explained 67% of the variance for HNI (disease rates formed their own factor, worth 46% of the index), and two factors explained 70% for HCI (licensed primary care formed its own factor). These numbers are historical and should be regenerated before being reported as current results under the updated inputs and transport assumptions. The share aged 60+ is negatively correlated with the share aged 0–4 and with poverty. Because indicators are direction-aligned before weighting, both age groups still add to need, consistent with the dependency-ratio rationale.

## 4b. Normalisation and aggregation (OECD & JRC, 2008)

1. Median-impute missing values and report the number imputed.
2. Log-transform skewed supply indicators.
3. Winsorise at the 1st and 99th percentiles.
4. Normalise to 0–100 (min–max; z-score with normal-CDF mapping as an alternative).
5. Average the indicators within each domain.
6. Aggregate domains with a weighted geometric mean, with each domain floored at 10, to limit compensability.

## 5. Healthcare Gap Index

- **Gap** = pct(HNI) − pct(HCI), in [−100, 100]; positive values mean need *ranks* higher than capacity. The indices are relative, so a negative gap does not mean adequate care; adequacy is judged in absolute terms against the DOH target (1.25 beds per 1,000) and the 2040 goal (4 per 1,000).
- **Bivariate class**: vulnerability tertile × capacity tertile; V3-C1 is the priority class.
- **Bed gap** = population/1000 × benchmark − accessible beds (benchmark: the DOH target of 1 bed per 800 people, i.e. 1.25 per 1,000, from the Philippine Health Facility Development Plan 2017–2022; the PHFDP 2020–2040 aspires to 4 beds per 1,000 by 2040).

## 6. Spatial statistics

- **Weights**: KNN with k = 6, row-standardised, which keeps island LGUs connected.
- **Global clustering**: Moran's I with 999 permutations.
- **Local clusters**: LISA (Anselin, 1995) with conditional permutation.
- **Hot spots**: Getis-Ord Gi* (Ord & Getis, 1995).
- **Multiple testing**: Benjamini–Hochberg FDR at 5% (Benjamini & Hochberg, 1995).
- **Typologies**: spatially constrained Ward clustering. SKATER (Assunção et al., 2006) via `spopt` is an alternative.

## 7. Robustness and validation

The Monte Carlo analysis runs 1,000 times. Each run draws:

- domain weights from a Dirichlet distribution (α = 10),
- the normalisation method (min–max or z-score),
- the catchment multiplier (0.5, 1.0, or 1.5).

For each LGU it reports the median rank, the 90% rank interval, and how often the LGU is classed as priority. LGUs whose priority status flips in more than 20% of runs are flagged as unstable.

For external validity, the pipeline computes Spearman correlations between the gap index and independent outcomes at LGU and province level, and checks that the sign matches expectations.

## 8. Limitations

- Data vintages differ across sources.
- Licensed beds are not the same as staffed or functional beds.
- Private-hospital cost barriers are not modelled.
- The modifiable areal unit problem: results can change with the unit of aggregation.
- The straight-line fallback is less accurate than network travel times.
- Results flag areas for investigation; they are not allocation decisions.


## 9. Equity analysis (Equity page)

LGUs are ranked by poverty incidence and grouped into population-weighted fifths, each holding about 20% of Filipinos. For each fifth, the page reports accessible beds per 1,000, the share of people within 30 minutes of a hospital and within 60 minutes of a Malasakit Center, the mean HGI, and the share of people living in priority LGUs. Concentration curves and the population-weighted concentration index summarise inequality by poverty: 0 means equality, and positive values mean the service favours less-poor LGUs. The same comparison is repeated for GIDA status and age structure.

## 10. Machine learning (ML Insights page; `isagiph/ml.py`)

**Expected capacity.** A regression predicts accessible beds per 1,000 (log scale) from poverty, GIDA share, age structure, density, and population size. The target is an E2SFCA-derived accessibility indicator; it is not an independent observation of staffed beds, patient utilisation, service quality, or a clinical outcome.
- **Models and baseline:** mean baseline, linear regression, Ridge regression, random forest (300 trees, minimum leaf size 5), and histogram-based gradient boosting, evaluated on identical folds.
- **Preprocessing:** median imputation and standardisation are fitted inside each training fold. Entirely missing features are excluded and documented; missing values are not globally imputed before validation.
- **Validation:** region-grouped outer GroupKFold holds complete regions outside each training fold. A nested grouped-CV pass estimates the model-family selection procedure and produces cross-fitted LGU expectations. The top single candidate ranked on pooled outer-fold scores is treated as exploratory because selecting it on those same scores can be optimistic. The primary model-vs-mean-baseline comparison uses nested predictions when available. If fewer than three regions are available, the system records that nested spatial selection could not be assessed and labels the fallback.
- **Metrics and uncertainty:** log-target R², MAE, RMSE and bias on the beds-per-1,000 scale; 95% confidence intervals obtained by resampling whole regions; subgroup error audits by region, GIDA-share stratum and population stratum.
- **Feature importance:** random-forest permutation importance is computed on held-out fold rows, with across-fold variability reported. Standardised linear coefficients are averaged across folds.
- **Prediction bands:** when nested regional validation is available, each outer-fold band is calibrated from inner out-of-fold residuals in that fold's training regions and assessed only on the corresponding held-out rows. If calibration is unavailable, the band is omitted rather than estimated from the same rows used to assess coverage. These empirical spatial-validation intervals are **not** formal distribution-free conformal intervals; residual spatial dependence and limited region counts may affect coverage. Coverage and interval widths are recorded in model metrics.
- **Outputs:** `data/processed/ml_predictions.csv`, `ml_importance.csv`, `ml_metrics.json`, `ml_region_metrics.csv`, `ml_subgroup_metrics.csv`, and `ml_validation.md`. `run_meta.json` also records data/config hashes and environment versions.

**Typology.** K-means on standardised domain scores. Clusters are named automatically from their centroids, for example "High need · low capacity (weakest: beds)"; k is selected by silhouette score and stability is checked by bootstrap refits. This is descriptive clustering, not a prediction target.

The older R² figures previously recorded in reports were produced by an earlier pipeline version. Re-run the pipeline after this update before citing current numeric performance.

## 11. ISAGIP Simulator (`isagiph/simulate.py`)

Users place hypothetical facilities on the map:
- **Hospitals:** added to the E2SFCA supply with their level and beds, with travel times from the same straight-line model.
- **Rural health units:** each adds one licensed primary care facility to the nearest LGU.

Access, both indices, and the gap are then recomputed for every LGU and compared with the baseline. Because the indices are national rankings, adding care in one place can slightly change other LGUs' scores; the impact panel therefore reports absolute measures (residents within 30 minutes, bed shortfall, people in priority LGUs) alongside the ranking changes.

## 12. Facility directory and community features

**Facility directory.** It lists all licensed DOH hospitals, infirmaries, and primary care facilities with their services. Hospital services come from the dialysis-station, blood-service, rHIV, clinical-lab, and ambulance columns and keyword matching of the add-on services; primary care services come from the service columns. Bed figures are licensed (authorised) capacity, not current occupancy.

**Community reports and the resource board.** Both are stored in local SQLite (`data/community/`). Reports are labelled unverified, can be flagged, and require a privacy confirmation. Photos are resized and re-encoded, which strips EXIF metadata including GPS location.
