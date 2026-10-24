# I-SAGIPH findings report

Generated 2026-10-24 14:27, mode: **real**, travel times: straight-line fallback (effective-speed assumption 15 km/h, circuity 1.4; configured transport assumptions for geographic context), sensitivity runs: 1000.

## 1. Headline numbers

- LGUs analysed: 1,634; population covered: 108,451,695.
- Priority LGUs (top Health Need Index tertile and bottom Hospital Capacity Index tertile): 292, home to 11,444,189 people; 187 remain priority in at least 80% of sensitivity runs.
- Accessible-bed shortfall against the DOH target of 1.25 beds per 1,000 (1 per 800): 50,091 beds nationally, 12,367 of them in priority LGUs.
- Beds within reach nationally: 1.12 per 1,000, below the DOH target of 1.25 and far below the 2040 goal of 4.0. No region reaches the 2040 goal; reaching it everywhere would need about 313,482 more beds within reach.
- **Modelled** residents with no hospital within 30 minutes: 41,457,947 (38.2%). This is an estimate under the configured transport assumptions, not an observed count of people unable to obtain care.

**Travel-time interpretation:** road-network estimates use configured road-class speeds plus a short access/egress speed assumption. Where a route is unavailable, the configured generalized effective-speed fallback is used to approximate mixed local travel. The reported number of residents beyond 30 minutes from a hospital is a model estimate; where a DOH reference figure is supplied, compare the values after estimation and check that their definitions and geographic scope align.

HNI = Health Need Index, HCI = Hospital Capacity Index, HGI = Healthcare Gap Index.

## 2. Is the gap spatially clustered?

Global Moran's I = 0.625 (E[I] = -0.0006, pseudo-p = 0.001, 999 permutations, KNN k = 6). Gaps cluster geographically, so responses can be organised regionally rather than LGU by LGU.

| LISA class (FDR 5%) | LGUs |
|---|---|
| Not significant | 1,077 |
| LL | 317 |
| HH | 213 |
| LH | 15 |
| HL | 12 |

| Gi* class (FDR 5%) | LGUs |
|---|---|
| Not significant | 1,148 |
| Cold spot | 270 |
| Hot spot | 216 |

## 3. Regional picture

| region | LGUs | priority LGUs | pop-weighted HGI | bed shortfall | % within 30 min |
|---|---|---|---|---|---|
| Region V (Bicol Region) | 114 | 56 | 37.0 | 4,050.2 | 45.2 |
| Mimaropa Region | 73 | 39 | 30.8 | 2,157.8 | 34.4 |
| Bangsamoro Autonomous Region In Muslim Mindanao (BARMM) | 119 | 52 | 20.9 | 3,411.2 | 44.3 |
| Region VIII (Eastern Visayas) | 143 | 20 | 11.7 | 3,148.2 | 36.8 |
| Region VI (Western Visayas) | 133 | 35 | 9.8 | 4,949.3 | 48.0 |
| Region IX (Zamboanga Peninsula) | 72 | 32 | 5.0 | 2,141.6 | 35.4 |
| Region XIII (Caraga) | 73 | 10 | -0.4 | 1,824.3 | 36.6 |
| Region XI (Davao Region) | 49 | 17 | -14.5 | 1,890.9 | 44.5 |
| Region X (Northern Mindanao) | 93 | 15 | -15.5 | 2,072.1 | 48.0 |
| Region II (Cagayan Valley) | 93 | 2 | -23.4 | 1,827.8 | 48.6 |
| Region I (Ilocos Region) | 125 | 4 | -26.6 | 2,456.5 | 66.2 |
| Cordillera Administrative Region (CAR) | 77 | 0 | -38.9 | 585.7 | 60.0 |
| Region VII (Central Visayas) | 132 | 1 | -48.5 | 4,652.2 | 53.1 |
| Region XII (Soccsksargen) | 49 | 0 | -49.4 | 1,723.6 | 44.7 |
| Region III (Central Luzon) | 130 | 2 | -56.9 | 5,549.6 | 75.9 |
| Region IV-A (Calabarzon) | 142 | 7 | -58.1 | 6,670.2 | 80.9 |
| National Capital Region (NCR) | 17 | 0 | -83.0 | 979.7 | 100.0 |

## 4. Highest-need LGUs (stable priority, ranked by bed shortfall)

| name | province | population | HNI | HCI | HGI | bed_gap_shortfall | cov_any_30 | rank_p05 | rank_p95 |
|---|---|---|---|---|---|---|---|---|---|
| Naujan | Oriental Mindoro | 109,492 | 52.89 | 10.00 | 60.69 | 123.40 | 0.23 | 166.95 | 286.00 |
| Bulan | Sorsogon | 104,964 | 65.35 | 10.00 | 86.96 | 113.13 | 0.54 | 14.00 | 227.00 |
| Murcia | Negros Occidental | 88,690 | 54.92 | 10.00 | 69.26 | 110.86 | 0.00 | 103.00 | 185.00 |
| Cauayan | Negros Occidental | 108,413 | 60.91 | 10.00 | 83.10 | 110.52 | 0.32 | 43.00 | 247.00 |
| Buhi | Camarines Sur | 81,277 | 59.22 | 10.00 | 80.10 | 101.60 | 0.00 | 13.00 | 78.00 |
| La Castellana | Negros Occidental | 79,412 | 59.47 | 10.03 | 68.40 | 99.27 | 0.00 | 61.00 | 160.00 |
| Aroroy | Masbate | 87,994 | 58.40 | 10.00 | 78.63 | 96.34 | 0.29 | 72.00 | 160.05 |
| Parang | Maguindanao del Norte | 102,830 | 54.69 | 10.00 | 68.28 | 95.72 | 0.44 | 163.00 | 416.00 |
| Pilar | Sorsogon | 75,758 | 63.99 | 10.00 | 86.53 | 92.08 | 0.10 | 16.00 | 38.00 |
| Bula | Camarines Sur | 73,088 | 60.01 | 10.00 | 81.63 | 91.26 | 0.06 | 20.00 | 74.00 |
| Carles | Iloilo | 72,626 | 56.29 | 10.23 | 61.54 | 83.83 | 0.09 | 98.00 | 222.00 |
| Talakag | Bukidnon | 77,027 | 53.92 | 10.00 | 65.09 | 82.26 | 0.28 | 122.95 | 239.00 |
| City of Escalante | Negros Occidental | 96,114 | 54.58 | 10.00 | 67.67 | 80.31 | 0.38 | 164.00 | 575.05 |
| Cawayan | Masbate | 69,104 | 53.97 | 10.00 | 65.46 | 76.38 | 0.11 | 93.00 | 221.00 |
| Bongabong | Oriental Mindoro | 76,876 | 53.99 | 10.00 | 65.52 | 76.00 | 0.35 | 164.00 | 263.00 |

## 5. External validation

No outcome indicators supplied.

A consistent result means the gap index moves in the expected direction with an outcome that was not used to build it.

## 6. Equity: who gets the beds?

Concentration index of beds within reach by poverty rank: **+0.157** (bootstrap 95% CI +0.102 to +0.205, 1,000 resamples of LGUs). The interval excludes 0: beds within reach are significantly concentrated in less-poor areas.

## 7. Bottlenecks and suggested actions

| bottleneck | areas | people | beds_short | action |
|---|---|---|---|---|
| Bed supply | 3 | 186,045 | 191 | Add beds where people already travel: expand or upgrade an existing hospital (Level 1 to Level 2) |
| Distance | 184 | 6,981,523 | 7,778 | Bring care closer: a new Level 1 hospital or infirmary, referral transport, or telemedicine |

Stable priority areas only. Distance = under half of residents within 30 min of a hospital; Bed supply = most residents within reach, but too few beds for them. See `reports/policy_brief.md` for the areas in each region.

- **Relative scores are not adequacy**: a low gap or high capacity score means better than other LGUs, not enough. Compare beds per 1,000 with the DOH target and the 2040 goal.
- **High vulnerability without Malasakit reach**: candidates for new Malasakit Centers (R.A. 11463).
- **Unstable LGUs**: results depend on modelling choices; verify with provincial health offices before acting.