# Data quality report (real data)

- Boundaries: 1,642 cities/municipalities, 42,048 barangays (COD-AB attributes).
- Census 2020: 1,634 cities/municipalities matched; 9 blocks unmatched (e.g. BARMM: INTERIM PROVINCE (Composed of 63 barangays from COTABATO), BARMM: CARMEN CLUSTER, BARMM: KABACAN CLUSTER, BARMM: MIDSAYAP CLUSTER I, BARMM: MIDSAYAP CLUSTER II, BARMM: PIGCAWAYAN CLUSTER, BARMM: PIKIT CLUSTER I, BARMM: PIKIT CLUSTER II).
- Poverty 2023 SAE: 1,596 LGUs (116 matched by name after code mismatch). Highly urbanized cities are not in this file and are imputed (see below).
- GIDA 2023: 6,242 GIDA barangays read; 6,231 linked to an LGU (11 unmatched); 953 LGUs have at least one GIDA barangay.
- Facility directory: 3,088 licensed facilities (1,330 hospitals, 603 infirmaries, 1,155 primary care); 19 not linked to an LGU; 1,829 located via OSM, 1,259 at their city/municipality's urban core (densest cluster of barangays), 0 at the LGU centroid; 138 of 146 PCSO Malasakit Centers linked; 661 list dialysis.
- Bedded facilities for access modelling: 20 specialty hospitals excluded from general bed access (e.g. the 4,200-bed National Center for Mental Health); infirmaries count as first-level beds.
- Disease rates (DOH FHSIS 2022, per 100,000): 197 areas per disease; 85 LGUs use their own city row, the rest their province's. Areas reported as 0 ('No Data') are imputed with the regional median: tb 266, pneumonia 178, hypertension 166, influenza 402, leptospirosis 979 LGUs.
- Primary care facilities (licensed): 1,159 records; 1,155 linked to an LGU.
- Free-standing ambulance providers: 134 records; 134 linked to an LGU.
- Excluded 8 LGUs with no 2020 census match (mostly created after 2020, e.g. Special Geographic Area - Carmen, Special Geographic Area - Kabacan, Special Geographic Area - Midsayap I, Special Geographic Area - Midsayap II, Special Geographic Area - Pigkawayan, Special Geographic Area - Pikit I).
- Poverty imputed with the regional median for 38 LGUs (mostly HUCs): Angeles City, Olongapo City, Lucena City (Capital), Iloilo City (Capital), Bacolod City (Capital), Cebu City (Capital), Lapu-Lapu City (Opon), Mandaue City, Tacloban City (Capital), Zamboanga City.... Flag: poverty_imputed.
- Missing disease rates (DOH 'No Data'): tb: regional median (0% lower held-out error than the regional median); pneumonia: regional median (0% lower held-out error than the regional median); hypertension: graph model (11% lower held-out error than the regional median).
- Demand points: 41,985 barangay centroids; LGU population split equally across barangays (barangay-level 2020 population would refine this).

**Final tables:** 1,634 LGUs, 41,985 demand points, 1,913 bedded facilities (121,624 beds; Level 2+: 536; Malasakit-linked: 119).

Disease rates are reported by DOH at province and city level, so municipalities share their province's rate.
- Travel-time assumptions: the road-routing module uses explicit road-class speed assumptions plus a 15 km/h access/egress assumption for short trips to/from mapped roads; where route data are unavailable, the fallback uses an effective speed of 15 km/h and circuity 1.4. These assumptions approximate mixed local transportation conditions; schedules, fares, transfers, waiting, and boat/ferry connections are not individually routed.
- Hospital access comparison: the pipeline estimates the population beyond a 30-minute hospital travel-time threshold from the facility inventory, population allocation, and configured transport assumptions. Where a DOH reference figure is supplied, it is compared with this estimate after calculation; check that scope and definitions align before interpreting differences.
- Map boundaries: official city/municipality outlines written to data/processed/lgu_boundaries.geojson.