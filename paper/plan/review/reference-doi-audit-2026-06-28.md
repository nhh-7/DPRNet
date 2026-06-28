# Reference DOI Audit -- 2026-06-28

## Summary

- Scope: all 45 entries in `paper/references.bib`.
- Method: parsed current BibTeX entries, queried CrossRef by DOI, compared title/year/pages against the local BibTeX metadata, and retained cite keys.
- Result: 45/45 DOIs returned CrossRef records; 45/45 title checks passed after correction; page fields now match CrossRef where CrossRef exposes page metadata.
- Corrections applied:
  - `dong2016fsrcnn`: corrected DOI from `10.1007/978-3-319-46475-6_24` to `10.1007/978-3-319-46475-6_25` and pages from `375--390` to `391--407`.
  - `wang2021survey`: corrected title to the DOI metadata title, `Deep Learning Algorithms for Single Image Super-Resolution: A Systematic Review`.
  - `zeyde2010set14`: corrected year from `2010` to the Springer/CrossRef publication year `2012`.
- Remaining note: `martin2001bsd100` returns title and pages from CrossRef but does not expose an issued year in the API response; the local `2001` year is retained because it matches the ICCV publication.

## Per-Entry Verification

| Key | DOI | CrossRef check | Action |
|---|---|---|---|
| `liu2025catanet` | `10.1109/CVPR52734.2025.01668` | title/year/pages match | verified |
| `ahn2018carn` | `10.1007/978-3-030-01249-6_16` | title/year/pages match | verified |
| `hui2019imdn` | `10.1145/3343031.3351084` | title/year/pages match | verified |
| `liu2020rfdn` | `10.1007/978-3-030-67070-2_2` | title/year/pages match | verified |
| `kong2022rlfn` | `10.1109/CVPRW56347.2022.00092` | title/year/pages match | verified |
| `liang2021swinir` | `10.1109/ICCVW54120.2021.00210` | title/year/pages match | verified |
| `zhang2022elan` | `10.1007/978-3-031-19790-1_39` | title/year/pages match | verified |
| `zhou2023srformer` | `10.1109/ICCV51070.2023.01174` | title/year/pages match | verified |
| `dong2014srcnn` | `10.1007/978-3-319-10593-2_13` | title/year/pages match | verified |
| `lim2017edsr` | `10.1109/CVPRW.2017.151` | title/year/pages match | verified |
| `zhang2024spin` | `10.1109/ICCV51070.2023.01169` | title/year/pages match; cite key kept stable although the venue year is 2023 | verified |
| `zhang2024atd` | `10.1109/CVPR52733.2024.00276` | title/year/pages match | verified |
| `yang2010sparse` | `10.1109/TIP.2010.2050625` | title/year/pages match | verified |
| `glasner2009single` | `10.1109/ICCV.2009.5459271` | title/year/pages match | verified |
| `kim2016vdsr` | `10.1109/CVPR.2016.182` | title/year/pages match | verified |
| `kim2016drcn` | `10.1109/CVPR.2016.181` | title/year/pages match | verified |
| `shi2016espcn` | `10.1109/CVPR.2016.207` | title/year/pages match | verified |
| `lai2017lapsrn` | `10.1109/CVPR.2017.618` | title/year/pages match | verified |
| `haris2018dbpn` | `10.1109/CVPR.2018.00179` | title/year/pages match | verified |
| `zhang2018rdn` | `10.1109/CVPR.2018.00262` | title/year/pages match | verified |
| `zhang2018rcan` | `10.1007/978-3-030-01234-2_18` | title/year/pages match | verified |
| `dong2016fsrcnn` | `10.1007/978-3-319-46475-6_25` | title/year/pages match after correction | fixed |
| `tai2017drrn` | `10.1109/CVPR.2017.298` | title/year/pages match | verified |
| `tai2017memnet` | `10.1109/ICCV.2017.486` | title/year/pages match | verified |
| `sun2022shufflemixer` | `10.52202/068431-1259` | title/year/pages match | verified |
| `guo2022dualregression` | `10.1109/TPAMI.2024.3406556` | title/year/pages match | verified |
| `wang2018nonlocal` | `10.1109/CVPR.2018.00813` | title/year/pages match | verified |
| `mei2020csnln` | `10.1109/CVPR42600.2020.00573` | title/year/pages match | verified |
| `yang2020ttn` | `10.1109/CVPR42600.2020.00583` | title/year/pages match | verified |
| `chen2021ipt` | `10.1109/CVPR46437.2021.01212` | title/year/pages match | verified |
| `liu2021swin` | `10.1109/ICCV48922.2021.00986` | title/year/pages match | verified |
| `chen2023hat` | `10.1109/CVPR52729.2023.02142` | title/year/pages match | verified |
| `zheng2023emt` | `10.1016/j.engappai.2024.108035` | title/year/pages match | verified |
| `wang2023omnisr` | `10.1109/CVPR52729.2023.02143` | title/year/pages match | verified |
| `choi2023ngswin` | `10.1109/CVPR52729.2023.00206` | title/year/pages match | verified |
| `liu2022hpinet` | `10.1609/aaai.v37i2.25254` | title/year/pages match | verified |
| `wang2021survey` | `10.3390/electronics10070867` | title/year/pages match after correction | fixed |
| `ren2024ntire` | `10.1109/CVPRW63382.2024.00656` | title/year/pages match | verified |
| `agustsson2017div2k` | `10.1109/CVPRW.2017.150` | title/year/pages match | verified |
| `bevilacqua2012set5` | `10.5244/C.26.135` | title/year/pages match | verified |
| `zeyde2010set14` | `10.1007/978-3-642-27413-8_47` | title/pages match; year corrected to CrossRef publication year | fixed |
| `martin2001bsd100` | `10.1109/ICCV.2001.937655` | title/pages match; CrossRef issued year unavailable | verified with note |
| `huang2015urban100` | `10.1109/CVPR.2015.7299156` | title/year/pages match | verified |
| `matsui2017manga109` | `10.1007/s11042-016-4020-z` | title/year/pages match | verified |
| `wang2004ssim` | `10.1109/TIP.2003.819861` | title/year/pages match | verified |

## Verification Commands

- Raw audit artifact: `paper/plan/review/reference-doi-audit-raw-2026-06-28.json`.
- Final CrossRef check summary: 45 entries, 0 CrossRef errors, 0 low-title-match entries, 0 page mismatches.
- DOI resolver `HEAD` responses were not used as the primary gate because several publishers return `202`, `403`, or rate-limit codes for automated `HEAD` requests even when CrossRef metadata resolves correctly.
