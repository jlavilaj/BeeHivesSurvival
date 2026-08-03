# BeeHivesSurvival

Data and analysis code supporting:

> **Evaluation of the honey bee brood humidity rearing protocol in laboratory conditions for ecotoxicological test in local bee subspecies _Apis mellifera iberiensis_**
> Felipe López-Villegas, José L. Ávila-Jiménez, Francisco J. Rodríguez-Lozano, José M. Flores.
> Submitted to *Applied Biosciences* (MDPI).

This repository accompanies the manuscript's Data Availability Statement. It contains a representative sample of the survival data recorded across the three laboratory assays described in the paper, together with the Python scripts used to run the Kaplan-Meier and Cox proportional-hazards survival analyses.

This repository is kept private prior to publication and will be made public when the article is published.

## Repository structure

```
BeeHivesSurvival/
├── data/
│   ├── raw/           Original Excel data sheets, unmodified
│   └── processed/     Cleaned CSV files used as input by the scripts
├── scripts/            Analysis scripts (one per assay / breakdown)
├── requirements.txt
├── LICENSE
└── README.md
```

## Study design (brief)

Three laboratory assays were used to evaluate the effect of rearing humidity on *A. m. iberiensis* brood survival:

- **Assay 1** – Effect of the two humidity regimes recommended for comparison, 95/80/60% (OECD standard) vs 70/70/60%, under the base laboratory protocol.
- **Assay 2** – Effect of transferring larvae to new cells lined with absorbent lab paper (from D9 onwards), evaluated within each humidity regime.
- **Assay 3** – Comparison of laboratory-reared brood (all humidity regime x transfer combinations) against brood survival recorded in the original hives, broken down by source colony.

Brood was checked on days D3, D4, D5, D6, D7, D8, D15 and D22 (D4 and D6 were not recorded in Assay 3). A larva/pupa is considered to have survived a given day if it is alive, mobile and shows no oedema or abnormal coloration; survival to D22 (adult emergence) is treated as the censoring point.

## Data files

| File | Rows | Description |
|---|---|---|
| `data/processed/assay1_humidity_regimes.csv` | 2160 | Assay 1: one row per larva, humidity regime (1 = 95/80/60, 2 = 70/70/60), and daily alive/dead status. |
| `data/processed/assay2_lab_paper_transfer.csv` | 1008 | Assay 2: as above, plus block (humidity regime) and transfer protocol (`transferred_code`: 1 = not transferred, 2 = transferred to new cells with lab paper). |
| `data/processed/assay3_lab_vs_hive.csv` | 1476 | Assay 3: as above, plus source colony and assay type (`assay_type_code`: 1 = laboratory, 2 = original hive). Observed on D3, D5, D8, D15, D22 only. |

Column reference:

| Column | Meaning |
|---|---|
| `run`, `plate`, `block` | Experimental run, culture plate, and humidity block identifiers. |
| `humidity_regime_code` | 1 = 95/80/60% (OECD standard), 2 = 70/70/60% (alternative regime). |
| `humidity_regime_raw` | Original text label recorded for the humidity regime (Assay 3 only). |
| `transferred_code` | 1 = not transferred, 2 = transferred to a new cell lined with absorbent lab paper (Assays 2 and 3). |
| `paper_label` | Original text label ("SIN PAPEL" / "CON PAPEL") corresponding to `transferred_code`. |
| `assay_type_code` / `assay_type_label` | 1 = laboratory-reared, 2 = original hive (Assay 3 only). |
| `colony` | Source colony identifier (Assay 3). |
| `larva_id` | Larva identifier within its plate. |
| `day_3` ... `day_22` | 1 if alive on that observation day, blank once deceased. |

The original, unmodified Excel workbooks are kept in `data/raw/` for provenance.

## Scripts

Each script is self-contained and can be run directly (`python scripts/<name>.py`). All of them read their input from `data/processed/` using a relative path, so they can be run from any location as long as the repository structure is preserved.

| Script | Reproduces |
|---|---|
| `survival_utils.py` | Shared helper that converts the daily alive/dead columns into `(time, event)` pairs for survival analysis. Imported by all other scripts. |
| `assay1_humidity_kaplan_meier_cox.py` | Kaplan-Meier curves and Cox model for Assay 1 (humidity regime comparison). |
| `assay2_lab_paper_transfer_kaplan_meier_cox.py` | Kaplan-Meier curves and Cox model for Assay 2, per humidity block (Figures 5A/5B). |
| `assay3_by_colony.py` | Assay 3, laboratory vs hive, per source colony (coarsest breakdown). |
| `assay3_by_colony_and_humidity.py` | Assay 3, hive vs the two humidity regimes, per source colony. |
| `assay3_by_colony_humidity_and_transfer.py` | Assay 3, full breakdown by humidity regime and transfer protocol, per source colony (feeds into manuscript Table 4). |
| `assay3_pooled_all_colonies.py` | Assay 3, pooled analysis across all colonies (Figure 6 and the overall Cox model reported in Section 3.3). |

Running each script prints the corresponding `lifelines` Cox model summary to the console and displays the survival plot(s). The reported hazard ratios, confidence intervals, concordance indices and likelihood-ratio tests have been checked to reproduce the values given in the manuscript and its Supplementary Materials.

## Requirements

```
pip install -r requirements.txt
```

Developed with Python 3.9+.

## Effect size calculation (Cohen's f)

The calculation of Cohen's f for Assay 3 (reported in the manuscript's Supplementary Materials) was performed directly from the Cox model's partial R² (derived from the likelihood-ratio test statistic) and is documented, with formulas, in the Supplementary Materials file accompanying the manuscript rather than as a separate script.

## Citation

If you use these data or scripts, please cite the manuscript above (full citation to be updated once a DOI is assigned).

## License

Code is released under the MIT License (see `LICENSE`). Data files are shared for the purpose of verifying and reproducing the results of the associated manuscript.
