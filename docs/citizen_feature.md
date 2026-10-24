# Para sa Mamamayan: citizen-facing service navigation

## Purpose

The `app/views/citizen.py` page complements the I-SAGIPH planning simulator. The simulator highlights modeled spatial access gaps; this page helps residents navigate the facility directory, find official medical-assistance information, prepare for a visit, and optionally report access barriers.

## Features

- Search the existing facility directory by name, service, LGU, province, region, and facility type.
- Filter for government facilities and facilities marked as Malasakit-linked in the project's dataset.
- View facility details and map positions, with a warning that coordinates may be city/municipality centres rather than exact entrances.
- Link directly to official DOH Malasakit Center directories, DSWD Malasakit information, the DOH medical-assistance guide, PhilHealth's accredited-facility directory, and the official emergency-hotline portal.
- Provide a visit-preparation checklist phrased as items to ask about, not universal requirements.
- Accept anonymous, optional reports of access barriers and summarize them by category and LGU. Individual free-text descriptions are not shown in the public aggregate view.

## Limits and responsible use

- The facility directory is built from source datasets and is not a live inventory of open services, staffed beds, or real-time availability.
- The build pipeline currently describes Malasakit Center matching against a PCSO list noted as November 2022. Therefore, the stored Malasakit flag is a research-dataset marker, not confirmation that a centre is operating today. Residents are directed to the current official DOH directories and should contact the facility before travelling.
- Assistance eligibility, documentary requirements, office hours, and available funding vary by program and case. I-SAGIPH does not process applications, determine eligibility, promise assistance, or replace advice from the facility's medical social service office.
- Barrier reports are community-submitted signals and are unverified. They are not official complaints or emergency alerts and are not automatically sent to government offices. Do not submit personal, identifying, or private medical information.
- Reports are stored locally in `data/community/` by the running instance. A public deployment needs an operator-approved privacy notice, retention/moderation processes, and appropriate access controls; ephemeral hosts may lose this data after restart.
- For emergencies, use the official Philippine emergency-hotline directory (including 911) or go to the nearest emergency facility. Do not use this page for urgent response.

## Official links embedded in the page

- DOH hospitals with Malasakit Centers: https://doh.gov.ph/mpo/mc-directory/doh
- LGU hospitals with Malasakit Centers: https://doh.gov.ph/mpo/mc-directory/lgu
- DSWD Malasakit Centers: https://crisis-intervention.dswd.gov.ph/malasakit-centers/
- DOH medical assistance guide (MAIFIP): https://doh.gov.ph/sites/default/files/Process-of-Availing-Medical-Assistance.pdf
- PhilHealth accredited facilities: https://www.philhealth.gov.ph/partners/providers/facilities/accredited/
- Official emergency hotlines: https://ehotlines.e.gov.ph/


## Community tabs

Within **Para sa Mamamayan**, the Community tab now contains **Access barriers**, **Facility reports**, and **Health announcements**. Facility reporting and the resource board reuse the existing Community and Resource Board workflows; these features are no longer separate top-level navigation sections.
