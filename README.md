# DS Classifier

Predicts the **dwelling structure** (`DWELLING_STRUCT`) of residential addresses from the geometry around them: the parcel (cadastre) an address sits on and the building footprints on that parcel. An XGBoost model is trained per state on addresses with a trusted (DCAN) dwelling structure, then used to score addresses whose structure is imputed, admin sourced or missing.

The class codes appear to follow the ABS Census dwelling structure classification. That reading is assumed, not defined in the code, so confirm it against your source data.

| Class | Meaning (ABS) |
|---|---|
| `x11` | Separate house |
| `x21`, `x22` | Semi-detached, row or terrace house, townhouse (one storey / two or more storeys) |
| `x31` – `x34` | Flat or apartment in a 1–2, 3, 4–8, or 9+ storey block |
| `x35`, `x91` – `x94` | Flat attached to a house, caravan, cabin or houseboat, improvised home, house or flat attached to a shop; merged into `OTHER` by default |

## How it works

```
addresses ─┐
           ├─► address ⇄ parcel linkage ───────────────┐
parcels  ──┤                                            ├─► training table ─► XGBoost ─► evaluation + candidate predictions
           ├─► building ⇄ parcel overlap + aggregates ──┤
buildings ─┘   (building_target: direct address ⇄ building attachment)
```

1. **Address ⇄ parcel linkage** (`generate_ar_cad_linkage`): points are joined to the parcel they fall in, and addresses outside every parcel go to the nearest one. Parcels under 100 m² are dropped. Parcels where more than six addresses each have their own geocode are written out for review (`mhe_for_review`).
2. **Building ⇄ parcel features** (`generate_bld_cad_linkage`): each building is overlapped with the parcels it intersects (intersection area, % of building on the parcel, distance to the boundary, boundary overlap, parcel crossing), and the results are summarised per parcel (counts, totals, means, spreads).
   - With `extended_features`, buildings also get nearest-neighbour distance, longest shared wall and a touching flag, and touching buildings are merged into one footprint first.
3. **Direct building attachment** (`method="building_target"`): on parcels where every address has a distinct geocode, each address within 5 m of a building is attached to that building. Those rows replace the parcel-level rows.
4. **Address density features**: addresses per parcel or building, building coverage, area per address, log transforms and similar.
5. **Training** (`train_test_model`): uses DCAN-sourced rows with a 90/10 stratified split. The model is XGBoost `hist` with per-state hyperparameters and round counts, plus early stopping. `MESHBLOCK` and `BLD_QUALITY_CLASS` are categorical features; their category lists are saved on the model so candidate rows are encoded the same way.
6. **Evaluation and prediction**: logs metrics, saves confusion matrix and per-class plots, writes validation predictions, then scores the candidate pool.

## Repository layout

```
dscm_run.ipynb                       Production driver notebook
functions/training_functions.py      The pipeline: loading, features, training, evaluation, prediction
scripts/generate_synthetic_data.py   Builds realistic synthetic inputs for local testing
scripts/run_pipeline.py              Command-line equivalent of the notebook
scripts/benchmark_pipeline.py        Speed and peak-memory benchmark (default 6M records)
scripts/generate_metrics_report.py   Renders a PDF metrics report from the prediction outputs
data/                                Local inputs and outputs (git-ignored)
```

## Setup

Python 3.12 with:

```
pandas geopandas numpy<=2.3 libpysal scikit-learn shapely xgboost matplotlib seaborn pyarrow
psutil         # benchmark only
reportlab      # PDF report only
```

With [uv](https://docs.astral.sh/uv/):

```
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe pandas geopandas "numpy<=2.3" libpysal scikit-learn shapely xgboost matplotlib seaborn pyarrow psutil reportlab
```

## Quick start (local, synthetic data)

```
.venv\Scripts\python scripts\generate_synthetic_data.py      # ~26k addresses for period a25, state SA
.venv\Scripts\python scripts\run_pipeline.py                 # full pipeline, same steps as the notebook
.venv\Scripts\python scripts\generate_metrics_report.py      # data/output/reports/a25_SA_metrics_report.pdf
```

The synthetic generator lays out a grid of dwelling arrangements (houses, semis, townhouse rows and blocks of flats of increasing size), with about 8% label noise. It also includes the edge cases the pipeline handles: addresses geocoded off-parcel, tiny easement parcels, buildings straddling parcel lines, touching buildings and duplicate ARIDs. Use `--records N` (or `--grid N`) to change its size.

## Options

Set these as variables in the notebook, or as flags on `run_pipeline.py` / `benchmark_pipeline.py`.

| Notebook variable | Script flag | Default | Effect |
|---|---|---|---|
| `period` | `--period` | `a25` | Data quarter; selects input files and output folders |
| `state` | `--state` | `SA` | State; also selects the hyperparameters and boosting rounds |
| `pool` | `--pool` | `all` | Address pool loaded (`training`, `candidate`, anything else = all residential) and candidates scored (`all`, `no_ds`, `dsam`) |
| `method` | `--method` | `building_target` | `building_target` adds direct address ⇄ building attachment; any other value uses parcel-level features only |
| `extended_features` | `--[no-]extended-features` | `True` | Building neighbour features and merging of touching buildings |
| `collapse_other_classes` | `--[no-]collapse-other-classes` | `True` | Merge `x35`, `x91`–`x94` into `OTHER`; `False` trains them as separate classes |
| `adhoc_output` | `--adhoc-output` | `None` | Write all outputs to this folder instead of the quarter folders (see below) |

### Ad hoc output (test runs in prod)

`tf.set_output_location("adhoc/2026-10-23_test")` (or `adhoc_output` in the notebook) redirects every output to that folder: the linkage, training table, predictions, plots and log. The training table that is read back for modelling also comes from there, so a test run never reads or overwrites a quarter's outputs. Inputs are unaffected. The path is relative to the output root, or absolute. Call it before `start_log`.

## Inputs

Paths are relative to `data/input/` locally (the S3 originals are kept as comments in the code).

| File | Key columns |
|---|---|
| `{period}_dsim/{period}_{training\|candidate\|all_res}_{state}.parquet` | `ARID`, `LATITUDE`, `LONGITUDE`, `MESHBLOCK`, `STATE_ABBREVIATION`, `DS_SOURCE`, `DWELLING_STRUCT` |
| `{period}_cad/{period}_cad_{state}.parquet` | `CADASTRE_POLYGON_PID`, `CADASTRE_PID`, `CAD_POLY` (WKT), `CAD_AREA`, `CAD_POLY_PID_COUNT`, `MB_CODE_2021` |
| `{period}_bld/{period}_bld_{state}.parquet` | `BUILDING_PID`, `BLD_POLY` (WKT), `BUILDING_AREA`, `QUALITY_CLASS`, `BUILDING_NUM_VERTICES`, `ADDRESS_COUNT`, `MB_CODE_2021` |

Geometries and coordinates are in GDA2020 (EPSG:7844). Metric calculations are done in GDA2020 Australian Albers (EPSG:9473).

`DS_SOURCE` determines how a row is used. Rows containing `DCAN` are training labels. For candidates, `IMPUTE` or a missing structure falls in the `all` pool, `ADMIN` in `dsam`, and an empty source in `no_ds`.

## Outputs

Paths are relative to `data/output/`, or to the ad hoc folder when one is set.

| Output | Path |
|---|---|
| Address ⇄ parcel linkage | `{period}/{state}/{state}_ar_cad_linkage.parquet` |
| Multi-address parcels for review | `{period}/{state}/{period}_{state}_mhe_for_review.csv` |
| Training table | `{period}_DSIM/{state}/{period}_{full_training\|full_candidate\|all_res_training}_{state}.parquet` |
| Validation predictions | `{period}_{state}_training_prediction_validation_output.csv` |
| Candidate predictions | `{period}_{state}_{pool}_predictions.csv` |
| Plots | `plots/confusion_matrix_{counts\|normalized}_{state}.png`, `plots/per_class_metrics_{state}.png` |
| Training log (full metrics) | `logs/model_training_log_{state}.txt` |

Prediction files have one probability column per class, plus `predicted_class`, `max_prob` and `ARID`. They also carry `true_class` (validation) or `original_class` (candidates).

## Metrics report

```
.venv\Scripts\python scripts\generate_metrics_report.py --period a25 --state SA --pool all [--adhoc-output ...] [--benchmark <json>]
```

The report is built from the prediction CSVs, so it needs neither the model nor a re-run. It covers:

- Headline metrics: accuracy, top-2 accuracy, macro and weighted F1, Cohen's kappa, Matthews correlation coefficient, log loss and expected calibration error.
- Per-class precision, recall and F1.
- The confusion matrix and the most common misclassifications.
- Calibration: how confident the model is compared with how often it's right.
- A candidate summary: predicted class mix against the existing classes, agreement and confidence bands.
- Optionally, a stage timing and memory table from a benchmark run.

## Benchmarking

```
.venv\Scripts\python scripts\benchmark_pipeline.py                         # 6,000,000 records
.venv\Scripts\python scripts\benchmark_pipeline.py --records 1000000 --num-rounds 50
```

The benchmark generates inputs in a separate process (under period `perf<records>`, apart from the test data). It then runs the full pipeline, timing every step and sampling peak memory, and writes a JSON and text report to `data/output/benchmarks/`. A memory guard (default 85% of RAM, or `--memory-limit-gb`) stops the run cleanly and still writes a partial report.

Measured on 1M records (50 rounds): 1.65 GB peak and about 5 minutes. Scaled up, 6M records needs about 9 GB peak, well within a 64 GB instance. Training time is about 2.6 s per boosting round per million rows; use `--num-rounds` to cap long runs.

## Running in production (S3)

The module reads and writes local `data/` paths, and each original S3 placeholder path is kept as a comment next to its local replacement.

- **Inputs:** swap the commented S3 line back in each loader (`load_ar_df`, `load_cad_gdf`, `load_bld_gdf`).
- **Outputs:** uncomment the single S3 line in `_output_path`. Every output, including ad hoc runs, goes through that function. Don't restore the S3 lines at individual call sites, because they bypass the ad hoc option.
- **Uploads:** plot and log uploads via boto3 are commented out in `_save_figure` and `predict_candidates`; restore them if those files should go to S3.
