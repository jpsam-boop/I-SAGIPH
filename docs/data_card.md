# Data card: sources, licences, and ethics

| Source | Level | Year | Use | Access / licence |
|---|---|---|---|---|
| 2020 CPH by barangay, age, sex (PSA via HDX) | Barangay | 2020 | Population, age structure, demand points | Public; cite PSA |
| Admin boundaries (COD-AB via HDX): attribute table and level-3 outlines | ADM0–ADM4 | Latest | Joins, centroids, map outlines | Public; cite OCHA/PSA/NAMRIA |
| DOH NHFR | Facility | Latest | Facility list, coordinates, primary care | Public web export |
| DOH licensed hospitals with beds (eFOI) | Hospital | Latest | Beds, level | eFOI release; cite request number |
| HOT OSM health facilities | Facility | Latest | Gap-filling coordinates | ODbL: attribution "© OpenStreetMap contributors" |
| PSA poverty incidence | City/mun. or province | Latest | Socioeconomic domain | Public |
| DOH FHSIS morbidity tables (TB, pneumonia, hypertension, influenza, leptospirosis) | Province/city | 2022 | Disease domain (first three scored) | Public; cite DOH |
| NDHS 2022 | Province/region | 2022 | **Validation only** | DHS registration; microdata **not redistributable** |
| DOH GIDA list | Municipality | Latest | Geographic domain | Public / request |
| Malasakit Center list | Hospital | Latest | Social protection domain | Public / request |

## Ethics and responsible use

- Only aggregate, publicly released data are used. The project processes no personal or patient-level records.
- NDHS microdata stay on team machines and are excluded from the repository by `.gitignore`. Only aggregated province-level indicators are shared.
- Facility information is shown as published by DOH and OSM. Errors are reported back to the source rather than silently corrected.
- Results flag areas for investigation and include uncertainty, so that LGUs are not stigmatised on the basis of noisy data.
- Every download is checksummed in `data/raw/MANIFEST.csv` to make provenance auditable.

## Community-contributed data

- **Community reports** (crowding, waiting time, affordability, optional photo), **resource board announcements**, and anonymous **access-barrier reports** are user-contributed and unverified. Access-barrier summaries are aggregated by category and LGU; they are not official complaints and are not automatically sent to government offices. Existing reports can be flagged and hidden.
- **Privacy:** users must confirm that photos and notes contain no faces, patients, or personal information. Photos are re-encoded without EXIF/GPS metadata.
- **Storage:** data are kept in `data/community/` on the server running the app and are excluded from the repository. Access-barrier reports avoid asking for personal or medical identifiers. Hosts with temporary storage, such as some free cloud tiers, will lose them on restart.
- **Production use:** a real deployment would need verified LGU accounts for announcements, moderation, and a privacy notice under the Data Privacy Act of 2012 (R.A. 10173).


## Travel-time assumptions and DOH context

Travel times are built from a Philippine roads GeoPackage where usable road-network coverage is available. The routing model uses road-class speed assumptions and a configured access/egress speed for short trips between mapped points and roads. Where routes are unavailable, fallback estimates use straight-line distance, a circuity factor and a configurable generalized effective speed intended to reflect mixed local travel patterns. This is not a mode-by-mode public-transport model: schedules, fares, waiting, transfers and boat/ferry legs are not individually represented.

Travel times use explicit transport assumptions reflecting road classes and generalized local/mixed travel where full routes are unavailable. DOH-licensed hospital and infirmary records supply the facility inventory used to estimate access; DOH access figures are used as an external comparison with the estimated population beyond the 30-minute hospital threshold, where definitions align. The PHFDP figure commonly cited as about 50% within 30 minutes refers to RHUs/health centres (primary care), not hospitals; it is contextual only and is not direct validation of the project's 30-minute hospital-access estimate.
