import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import xgboost as xgb
# import boto3  # only used for the S3 uploads, which are commented out for local runs
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Local stand-ins for the S3 locations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_INPUT_DIR = PROJECT_ROOT / "data" / "input"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

SOURCE_CRS = "EPSG:7844"   # GDA2020 geographic, as stored in the source WKT / lat-long
METRIC_CRS = "EPSG:9473"   # GDA2020 Australian Albers, for areas, lengths and distances

PID = 'CADASTRE_POLYGON_PID'
CATEGORICAL_FEATURES = ['MESHBLOCK', 'BLD_QUALITY_CLASS']
NON_FEATURE_COLUMNS = ['STATE_ABBREVIATION', 'CADASTRE_POLYGON_PID', 'CADASTRE_PID', 'coord']
OTHER_DWELLING_CODES = ['x35', 'x91', 'x92', 'x93', 'x94']


# Ad hoc output folder (see set_output_location); None means the standard quarter-based layout
_ADHOC_OUTPUT = None


def set_output_location(adhoc_output=None):
    """Redirect every output to an ad hoc folder instead of the quarter-based folders, e.g. for test runs in prod.

    `adhoc_output` is a folder relative to the output root (e.g. "adhoc/2026-10-23_test") or an absolute path.
    It replaces the {period}/{state} and {period}_DSIM/{state} folders, and also holds the plots, logs and
    prediction files, so a test run never overwrites quarter outputs. The training table that train_test_model
    and predict_candidates read back comes from the same folder. Inputs are unaffected. Call before start_log;
    pass None to restore the standard layout.
    """
    global _ADHOC_OUTPUT
    _ADHOC_OUTPUT = str(adhoc_output).strip("/") if adhoc_output else None
    if _ADHOC_OUTPUT:
        print(f"Writing outputs to ad hoc location: {_output_path('')}")


def _output_path(file_name, quarter_folder=""):
    """Where an output lives: under `quarter_folder` normally, or under the ad hoc folder when one is set."""
    folder = _ADHOC_OUTPUT if _ADHOC_OUTPUT is not None else quarter_folder
    # Prod equivalent - one change here routes every output, ad hoc included:
    # return f"s3://PLACEHOLDER_OUTPUT_PATH/{folder + '/' if folder else ''}{file_name}"
    path = LOCAL_OUTPUT_DIR / folder / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _log_time(label, start_time):
    message = f"{label}Execution Time: {(time.perf_counter() - start_time):.4f} seconds"
    print(message)
    logging.info(message)


def start_log(state):
    logging.basicConfig(
        filename=_output_path(f"logs/model_training_log_{state}.txt"),
        level=logging.INFO,
        format="%(message)s",
        filemode="w",
        force=True
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _wkt_to_gdf(df, wkt_column):
    # pop() drops the WKT strings as soon as they are parsed - they are the largest column by far
    geometry = gpd.GeoSeries.from_wkt(df.pop(wkt_column), crs=SOURCE_CRS)
    return gpd.GeoDataFrame(df, geometry=geometry).to_crs(METRIC_CRS)


def load_ar_df(period, state, pool):
    file_name = {
        "training": f"{period}_training_{state}.parquet",
        "candidate": f"{period}_candidate_{state}.parquet",
    }.get(pool, f"{period}_all_res_{state}.parquet")
    # ar_df = pd.read_parquet(f"s3://DATLAKE_PATH_PLACEHOLDER/{period}_dsim/{file_name}")
    ar_df = pd.read_parquet(LOCAL_INPUT_DIR / f"{period}_dsim" / file_name)
    return ar_df.groupby('ARID', as_index=False).first()


def load_cad_gdf(period, state):
    # cad_df = pd.read_parquet(f"s3://DATLAKE_PATH_PLACEHOLDER/{period}_cad/{period}_cad_{state}.parquet").drop(columns="MB_CODE_2021")
    cad_df = pd.read_parquet(LOCAL_INPUT_DIR / f"{period}_cad" / f"{period}_cad_{state}.parquet").drop(columns="MB_CODE_2021")
    return _wkt_to_gdf(cad_df, 'CAD_POLY')


def load_bld_gdf(period, state):
    # bld_df = pd.read_parquet(f's3://PLACEHOLDER_FILE_PATH/{period}_bld/{period}_bld_{state}.parquet').drop(columns='MB_CODE_2021')
    bld_df = pd.read_parquet(LOCAL_INPUT_DIR / f"{period}_bld" / f"{period}_bld_{state}.parquet").drop(columns='MB_CODE_2021')
    return _wkt_to_gdf(bld_df, 'BLD_POLY')


# ---------------------------------------------------------------------------
# Feature building blocks
# ---------------------------------------------------------------------------

def _distinct_geocode_parcels(ar_cad, min_count):
    """PIDs where every address has its own geocode and there are more than `min_count` of them."""
    row_count = ar_cad[PID].value_counts()
    distinct_count = (
        ar_cad[[PID, 'LATITUDE', 'LONGITUDE']].dropna().drop_duplicates()[PID]
        .value_counts()
        .reindex(row_count.index, fill_value=0)
    )
    return row_count.index[(row_count == distinct_count) & (distinct_count > min_count)]


def _add_neighbour_features(bld_gdf):
    """Nearest other building distance, longest shared wall and touch flag for each building."""
    geoms = bld_gdf[['geometry']]
    nearest = gpd.sjoin_nearest(geoms, geoms, how='inner', distance_col='distance', exclusive=True)
    neighbour_geoms = geoms.geometry.loc[nearest['index_right']].array
    nearest['shared_length'] = shapely.length(
        shapely.intersection(shapely.boundary(nearest.geometry.array), shapely.boundary(neighbour_geoms))
    )
    per_building = nearest.groupby(level=0)
    nearest_distance = per_building['distance'].min()
    return bld_gdf.assign(
        BLD_MAX_SHARED_LENGTH=per_building['shared_length'].max(),
        BLD_NEAREST_DISTANCE=nearest_distance,
        BLD_TOUCH=(nearest_distance.reindex(bld_gdf.index) < 0.001).astype(int)
    )


def _dissolve_touching_buildings(bld_gdf, aggfunc):
    """Merge buildings that share a vertex or edge (queen contiguity) into a single footprint."""
    W = libpysal.weights.Queen.from_dataframe(bld_gdf, use_index=False, silence_warnings=True)
    return bld_gdf.dissolve(by=W.component_labels, aggfunc=aggfunc)


def _building_parcel_overlap(bld_gdf, cad_gdf):
    """One row per (building, intersecting parcel) with overlap metrics. Both inputs in METRIC_CRS."""
    joined = gpd.sjoin(bld_gdf, cad_gdf[[PID, 'geometry']], how='inner', predicate='intersects')
    bld_geoms = joined.geometry.array
    cad_geoms = cad_gdf.geometry.loc[joined.pop('index_right')].array
    cad_boundaries = shapely.boundary(cad_geoms)
    joined['BLD_PERIMETER'] = shapely.length(bld_geoms)
    joined['BLD_INTERSECTION_AREA'] = shapely.area(shapely.intersection(bld_geoms, cad_geoms))
    joined['PCT_OF_BLD_INTERSECT'] = (joined['BLD_INTERSECTION_AREA'] / joined['BUILDING_AREA']) * 100
    joined['BLD_CROSS_PARCEL'] = (joined['PCT_OF_BLD_INTERSECT'] < 100).astype(int)
    joined['BLD_CAD_DISTANCE'] = shapely.distance(bld_geoms, cad_boundaries)
    joined['BLD_CAD_BOUNDARY_INTERSECTION'] = shapely.length(shapely.intersection(bld_geoms, cad_boundaries))
    return pd.DataFrame(joined.drop(columns='geometry'))


# Output column -> (input column, aggregation). Order matches the original flattened column lists.
_BLD_AGG_BASE = {
    'NUM_BUILDINGS': ('BUILDING_AREA', 'count'),
    'TOTAL_BUILDING_AREA': ('BUILDING_AREA', 'sum'),
    'MEAN_BUILDING_AREA': ('BUILDING_AREA', 'mean'),
    'STD_BUILDING_AREA': ('BUILDING_AREA', 'std'),
    'MIN_BUILDING_AREA': ('BUILDING_AREA', 'min'),
    'MAX_BUILDING_AREA': ('BUILDING_AREA', 'max'),
    'MEAN_BUILDING_PERIMETER': ('BLD_PERIMETER', 'mean'),
    'TOTAL_BUILDING_PERIMETER': ('BLD_PERIMETER', 'sum'),
    'MIN_BUILDING_INTERSECTION_AREA': ('BLD_INTERSECTION_AREA', 'min'),
    'MEAN_BUILDING_INTERSECTION_AREA': ('BLD_INTERSECTION_AREA', 'mean'),
    'MAX_BUILDING_INTERSECTION_AREA': ('BLD_INTERSECTION_AREA', 'max'),
    'TOTAL_BUILDING_INTERSECTION_AREA': ('BLD_INTERSECTION_AREA', 'sum'),
    'MIN_PCT_BUILDING_INTERSECTION': ('PCT_OF_BLD_INTERSECT', 'min'),
    'MEAN_PCT_BUILDING_INTERSECTION': ('PCT_OF_BLD_INTERSECT', 'mean'),
    'MAX_PCT_BUILDING_INTERSECTION': ('PCT_OF_BLD_INTERSECT', 'max'),
    'TOTAL_BLD_ADDRESS_COUNT': ('ADDRESS_COUNT', 'sum'),
    'BLD_QUALITY_CLASS': ('QUALITY_CLASS', 'first'),
}
_BLD_AGG_EXTENDED = {
    'MIN_BLD_NEAREST_DISTANCE': ('BLD_NEAREST_DISTANCE', 'min'),
    'MIN_BLD_MAX_SHARED_LENGTH': ('BLD_MAX_SHARED_LENGTH', 'min'),
    'MEAN_BLD_MAX_SHARED_LENGTH': ('BLD_MAX_SHARED_LENGTH', 'mean'),
    'MAX_BLD_MAX_SHARED_LENGTH': ('BLD_MAX_SHARED_LENGTH', 'max'),
    'SUM_BLD_MAX_SHARED_LENGTH': ('BLD_MAX_SHARED_LENGTH', 'sum'),
    'SUM_BLD_TOUCHING': ('BLD_TOUCH', 'sum'),
}
_BLD_AGG_CAD = {
    'MIN_BLD_CAD_DISTANCE': ('BLD_CAD_DISTANCE', 'min'),
    'MEAN_BLD_CAD_DISTANCE': ('BLD_CAD_DISTANCE', 'mean'),
    'MAX_BLD_CAD_DISTANCE': ('BLD_CAD_DISTANCE', 'max'),
    'SUM_BLD_CAD_DISTANCE': ('BLD_CAD_DISTANCE', 'sum'),
    'MIN_BLD_CAD_BOUNDARY_INTERSECTION': ('BLD_CAD_BOUNDARY_INTERSECTION', 'min'),
    'MEAN_BLD_CAD_BOUNDARY_INTERSECTION': ('BLD_CAD_BOUNDARY_INTERSECTION', 'mean'),
    'MAX_BLD_CAD_BOUNDARY_INTERSECTION': ('BLD_CAD_BOUNDARY_INTERSECTION', 'max'),
    'SUM_BLD_CAD_BOUNDARY_INTERSECTION': ('BLD_CAD_BOUNDARY_INTERSECTION', 'sum'),
    'SUM_BLD_CROSS_PARCEL': ('BLD_CROSS_PARCEL', 'sum'),
}


def _aggregate_buildings_per_parcel(overlap, extended_features=False):
    spec = {**_BLD_AGG_BASE, **(_BLD_AGG_EXTENDED if extended_features else {}), **_BLD_AGG_CAD}
    building_agg = overlap.groupby(PID).agg(**spec).reset_index()
    building_agg['BUILDING_SIZE_CV'] = (
        building_agg['STD_BUILDING_AREA'] /
        (building_agg['MEAN_BUILDING_AREA'] + 1e-6)
    )
    building_agg['BLD_SIZE_RANGE_RATIO'] = (
        building_agg['MAX_BUILDING_AREA'] /
        (building_agg['MIN_BUILDING_AREA'] + 1e-6)
    )
    return building_agg


def _add_address_ratio_features(df, group_column):
    """Address density features, with addresses counted per `group_column` (parcel or building)."""
    df['ARIDS_PER_PARCEL'] = df.groupby(group_column)['ARID'].transform('nunique')
    df['ARIDS_PER_BUILDING_RATIO'] = df['ARIDS_PER_PARCEL'] / df['NUM_BUILDINGS']
    df['BUILDING_AREA_PER_ARID'] = df['TOTAL_BUILDING_AREA'] / df['ARIDS_PER_PARCEL']
    df['CAD_AREA_PER_ARID'] = df['CAD_AREA'] / df['ARIDS_PER_PARCEL']
    df['ARID_DENSITY'] = df['ARIDS_PER_PARCEL'] / df['CAD_AREA']
    df['BUILDING_COVERAGE_RATIO'] = df['TOTAL_BUILDING_AREA'] / df['CAD_AREA']
    df['LOG_CAD_AREA'] = np.log1p(df['CAD_AREA'])
    df['LOG_BUILDING_TOTAL_AREA'] = np.log1p(df['TOTAL_BUILDING_AREA'])
    df['LOG_ARID_PER_PARCEL'] = np.log1p(df['ARIDS_PER_PARCEL'])
    df['BUILDING_TO_CAD_SIZE_RATIO'] = df['MEAN_BUILDING_AREA'] / df['CAD_AREA']
    return df


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

# GENERATE ADR_CAD_LINKAGE_TRAINING
def generate_ar_cad_linkage(period, state, pool="candidate", cad_gdf=None):
    start_time = time.perf_counter()
    print("Generating ADDR CAD Linkage")
    logging.info("Generating ADDR CAD Linkage: Loading files")
    ar_df = load_ar_df(period, state, pool)
    if cad_gdf is None:
        cad_gdf = load_cad_gdf(period, state)
    cad_gdf = cad_gdf[cad_gdf['CAD_AREA'] > 100]
    cad_perimeter = cad_gdf.geometry.length
    cad_gdf = cad_gdf.assign(
        CAD_PERIMETER=cad_perimeter,
        CAD_COMPACTNESS=4 * np.pi * cad_gdf['CAD_AREA'] / (cad_perimeter ** 2 + 1e-6)
    )

    ar_gdf = gpd.GeoDataFrame(
        ar_df,
        geometry=gpd.points_from_xy(ar_df.LONGITUDE, ar_df.LATITUDE, crs=SOURCE_CRS)
    ).to_crs(METRIC_CRS)
    del ar_df
    logging.info("Generating ADDR CAD Linkage: Computing join")
    joined = gpd.sjoin(ar_gdf, cad_gdf, how='left', predicate='intersects')
    unmatched = joined[PID].isna()
    # Addresses outside every parcel are attached to the closest one
    joined_nearest = gpd.sjoin_nearest(ar_gdf.loc[joined.index[unmatched]], cad_gdf, how='inner', distance_col='dist')
    joined_nearest = joined_nearest.sort_values('dist').drop_duplicates(subset='ARID', keep='first').drop(columns='dist')
    del ar_gdf, cad_gdf
    ar_cad = pd.DataFrame(
        pd.concat([joined[~unmatched], joined_nearest]).drop(columns=['geometry', 'index_right'])
    )
    del joined, joined_nearest

    ar_mhe_for_review = ar_cad.loc[
        ar_cad[PID].isin(_distinct_geocode_parcels(ar_cad, min_count=6)), ['ARID', PID]
    ]
    _log_time("", start_time)
    return ar_cad, ar_mhe_for_review


def generate_bld_cad_linkage(period, state, extended_features=False, bld_gdf=None, cad_gdf=None):
    start_time = time.perf_counter()
    logging.info("Loading building and cadastre files")
    if bld_gdf is None:
        bld_gdf = load_bld_gdf(period, state)
    if cad_gdf is None:
        cad_gdf = load_cad_gdf(period, state)
    if extended_features:
        bld_gdf = _add_neighbour_features(bld_gdf)
        bld_gdf = _dissolve_touching_buildings(bld_gdf, aggfunc={
            'BUILDING_AREA': 'sum',
            'QUALITY_CLASS': 'first',
            'BLD_MAX_SHARED_LENGTH': 'max',
            'BLD_TOUCH': 'sum',
            'BLD_NEAREST_DISTANCE': 'min',
            'BUILDING_NUM_VERTICES': 'sum',
            'ADDRESS_COUNT': 'max'
        })
    logging.info('Computing building/parcel overlap...')
    overlap = _building_parcel_overlap(bld_gdf, cad_gdf)
    overlap = overlap[overlap['PCT_OF_BLD_INTERSECT'] > 10]
    building_agg = _aggregate_buildings_per_parcel(overlap, extended_features)
    _log_time("", start_time)
    return building_agg


def generate_ar_building_attachment(ar_cad, period, state, pool, bld_gdf=None, cad_gdf=None):
    # Where there is the same number of residential ADDR and distinct geocodes on a parcel, these are direct candidates for joining to specific buildings
    direct_building_join_cand = ar_cad[ar_cad[PID].isin(_distinct_geocode_parcels(ar_cad, min_count=1))]
    if bld_gdf is None:
        bld_gdf = load_bld_gdf(period, state)
    if cad_gdf is None:
        cad_gdf = load_cad_gdf(period, state)
    bld_gdf = _dissolve_touching_buildings(bld_gdf, aggfunc={
        'BUILDING_PID': 'first',
        'BUILDING_AREA': 'sum',
        'QUALITY_CLASS': 'first',
        'BUILDING_NUM_VERTICES': 'sum',
        'ADDRESS_COUNT': 'max'
    })
    bld_gdf['NEW_ID'] = np.arange(len(bld_gdf))

    ar_gdf = gpd.GeoDataFrame(
        direct_building_join_cand,
        geometry=gpd.points_from_xy(direct_building_join_cand.LONGITUDE, direct_building_join_cand.LATITUDE, crs=SOURCE_CRS)
    ).to_crs(METRIC_CRS)
    nearest = gpd.sjoin_nearest(ar_gdf, bld_gdf[['NEW_ID', 'geometry']], how='inner', distance_col='distance')
    acceptable_distance = pd.DataFrame(
        nearest.loc[nearest['distance'] < 5].drop(columns=['geometry', 'index_right', 'distance'])
    )
    del ar_gdf, nearest

    logging.info("Computing building/parcel overlap for directly attached addresses...")
    overlap = _building_parcel_overlap(bld_gdf, cad_gdf)
    overlap = overlap[(overlap['PCT_OF_BLD_INTERSECT'] > 20) | (overlap['PCT_OF_BLD_INTERSECT'] == 0)]
    overlap = overlap[overlap[PID].isin(acceptable_distance[PID])]
    building_agg = _aggregate_buildings_per_parcel(overlap)

    joined_df = acceptable_distance.merge(building_agg, how='left', on=PID).drop_duplicates(subset='ARID')
    joined_df = _add_address_ratio_features(joined_df, group_column='NEW_ID')
    return joined_df.drop(columns='NEW_ID')


def _training_output_path(period, state, pool):
    file_name = {
        "training": f"{period}_full_training_{state}.parquet",
        "candidate": f"{period}_full_candidate_{state}.parquet",
    }.get(pool, f"{period}_all_res_training_{state}.parquet")
    # return f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{file_name}"
    return _output_path(file_name, f"{period}_DSIM/{state}")


def generate_training_data_workflow(period, state, pool, method, extended_features):
    init_time = time.perf_counter()
    # Parsed once and shared by every linkage step
    cad_gdf = load_cad_gdf(period, state)
    ar_cad, ar_mhe_for_review = generate_ar_cad_linkage(period, state, pool, cad_gdf=cad_gdf)
    # ar_cad.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}/{state}/{state}_ar_cad_linkage.parquet", index=False)
    ar_cad.to_parquet(_output_path(f"{state}_ar_cad_linkage.parquet", f"{period}/{state}"), index=False)
    # ar_mhe_for_review.to_csv(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}/{state}/{period}_{state}_mhe_for_review.csv", index=False)
    ar_mhe_for_review.to_csv(_output_path(f"{period}_{state}_mhe_for_review.csv", f"{period}/{state}"), index=False)
    del ar_mhe_for_review

    bld_gdf = load_bld_gdf(period, state)
    bld_cad = generate_bld_cad_linkage(period, state, extended_features, bld_gdf=bld_gdf, cad_gdf=cad_gdf)
    joined_df = ar_cad.merge(bld_cad, how='left', on=PID).drop_duplicates(subset='ARID')
    del bld_cad
    joined_df = _add_address_ratio_features(joined_df, group_column=PID)

    if method == "building_target":
        ar_bld = generate_ar_building_attachment(ar_cad, period, state, pool, bld_gdf=bld_gdf, cad_gdf=cad_gdf)
        joined_df = joined_df[~joined_df['ARID'].isin(ar_bld['ARID'])]
        if extended_features:
            # Building-attached rows have no neighbour features, so drop them for the whole pool
            joined_df = joined_df.drop(columns=list(_BLD_AGG_EXTENDED))
        joined_df = pd.concat([joined_df, ar_bld], ignore_index=True)
        del ar_bld
    del ar_cad, bld_gdf, cad_gdf

    joined_df.to_parquet(_training_output_path(period, state, pool), index=False)
    _log_time(f"Completed {state}. ", init_time)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _default_class_names(n):
    return [f"Class {i}" for i in range(n)]


def _predict_proba(model, X):
    return model.inplace_predict(X)


def get_confusion_matrix(model, X_test, y_test):
    predictions = _predict_proba(model, X_test).argmax(axis=1)
    cm = confusion_matrix(y_test, predictions)
    return cm, predictions


def plot_confusion_matrix(cm, class_names=None, normalize=False, figsize=(10, 8)):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'
    if class_names is None:
        class_names = _default_class_names(len(cm))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax
    )
    # First class at the bottom, so the correct-prediction diagonal runs bottom-left to top-right
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    fig.tight_layout()
    return fig


def calculate_all_metrics(y_test, predictions, class_names=None):
    metrics = {'accuracy': accuracy_score(y_test, predictions)}
    for average in ('macro', 'weighted'):
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=average, zero_division=0)
        metrics[f'precision_{average}'] = precision
        metrics[f'recall_{average}'] = recall
        metrics[f'f1_{average}'] = f1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None, zero_division=0)
    metrics['precision_per_class'] = precision
    metrics['recall_per_class'] = recall
    metrics['f1_per_class'] = f1
    metrics['cohen_kappa'] = cohen_kappa_score(y_test, predictions)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_test, predictions)
    return metrics


def print_metrics_summary(metrics, class_names=None):
    logging.info("\n" + "=" * 70)
    logging.info("MODEL PERFORMANCE METRICS - TEST")
    logging.info("=" * 70)
    logging.info("OVERALL METRICS")
    logging.info("=" * 70)
    logging.info(f" Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logging.info(f" Precision (macro): {metrics['precision_macro']:.4f}")
    logging.info(f" Precision (weighted): {metrics['precision_weighted']:.4f}")
    logging.info(f" Recall (macro): {metrics['recall_macro']:.4f}")
    logging.info(f" Recall (weighted): {metrics['recall_weighted']:.4f}")
    logging.info(f" F1-Score (macro): {metrics['f1_macro']:.4f}")
    logging.info(f" F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    logging.info(f" Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    logging.info(f" Matthews Corr. Coef: {metrics['matthews_corrcoef']:.4f}")

    logging.info("\n PER-CLASS METRICS")
    logging.info("-" * 70)
    if class_names is None:
        class_names = _default_class_names(len(metrics['precision_per_class']))
    logging.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logging.info("-" * 70)
    for name, prec, rec, f1 in zip(class_names, metrics['precision_per_class'], metrics['recall_per_class'], metrics['f1_per_class']):
        logging.info(f"{name:<20} {prec:>11.4f} {rec:>11.4f} {f1:>11.4f}")
    logging.info("=" * 70)


def get_classification_report(y_test, predictions, class_names=None):
    target_names = class_names if class_names is not None else _default_class_names(len(np.unique(y_test)))
    return classification_report(
        y_test,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0
    )


def analyze_per_class_performance(cm, class_names=None):
    if class_names is None:
        class_names = _default_class_names(len(cm))

    logging.info("\n" + "=" * 70)
    logging.info("PER-CLASS DETAILED ANALYSIS")
    logging.info("=" * 70)
    total = cm.sum()
    for i, name in enumerate(class_names):
        logging.info(f"\n{name}:")
        logging.info("-" * 70)
        total_actual = cm[i, :].sum()
        total_predicted = cm[:, i].sum()
        tp = cm[i, i]
        fp = total_predicted - tp
        fn = total_actual - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logging.info(f" True Positives:     {tp:>6g}")
        logging.info(f" False Positives:    {fp:>6g}")
        logging.info(f" False Negatives:    {fn:>6g}")
        logging.info(f" True Negatives:     {tn:>6g}")
        logging.info(f" Total Actual:       {total_actual:>6g}")
        logging.info(f" Total Predicted:    {total_predicted:>6g}")
        logging.info(f" Precision:          {precision:>6.4f}")
        logging.info(f" Recall:             {recall:>6.4f}")
        logging.info(f" F1-Score:           {f1:>6.4f}")


def plot_per_class_metrics(metrics, class_names=None, figsize=(12, 6)):
    if class_names is None:
        class_names = _default_class_names(len(metrics['precision_per_class']))

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics['recall_per_class'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    fig.tight_layout()
    return fig


def misclassification_analysis(y_test, predictions, class_names=None, cm=None):
    if cm is None:
        cm = confusion_matrix(y_test, predictions)
    if class_names is None:
        class_names = _default_class_names(len(cm))
    logging.info("\n" + "=" * 70)
    logging.info("MISCLASSIFICATION ANALYSIS")
    logging.info("=" * 70)

    true_idx, pred_idx = np.nonzero(cm * (1 - np.eye(len(cm), dtype=cm.dtype)))
    misclass = sorted(
        (
            {'true': class_names[i], 'predicted': class_names[j], 'count': cm[i, j], 'true_idx': i, 'pred_idx': j}
            for i, j in zip(true_idx, pred_idx)
        ),
        key=lambda item: item['count'],
        reverse=True
    )
    logging.info('\n Top 10 Misclassification Patterns:')
    logging.info('-' * 70)
    logging.info(f"{'True Class':<20} -> {'Predicted':<20} {'Count':>10}")
    logging.info('-' * 70)
    for item in misclass[:10]:
        logging.info(f"{item['true']:<20} -> {item['predicted']:<20} {item['count']:>10}")
    return misclass


def _save_figure(fig, file_name):
    # Previously written to the working directory, uploaded to S3 and then deleted:
    # s3_client = boto3.client('s3')
    # s3_client.upload_file(Filename=file_name, Bucket="PLACEHOLDER", Key="PLACEHOLDER")
    fig.savefig(_output_path(f"plots/{file_name}"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def complete_model_evaluation(period, state, method, model, X_test, y_test, class_names=None):
    logging.info("\n" + "=" * 70)
    logging.info("COMPLETE MODEL EVALUATION")
    logging.info("=" * 70)

    logging.info("\n[1] Making predictions on test set...")
    predictions = _predict_proba(model, X_test)
    class_predictions = predictions.argmax(axis=1)
    y_test = np.asarray(y_test).ravel()

    cm = confusion_matrix(y_test, class_predictions)
    metrics = calculate_all_metrics(y_test, class_predictions, class_names)
    print_metrics_summary(metrics, class_names)

    report = get_classification_report(y_test, class_predictions, class_names)
    logging.info("\n" + report)

    analyze_per_class_performance(cm, class_names)
    misclass = misclassification_analysis(y_test, class_predictions, class_names, cm=cm)

    _save_figure(plot_confusion_matrix(cm, class_names, normalize=False), f'confusion_matrix_counts_{state}.png')
    _save_figure(plot_confusion_matrix(cm, class_names, normalize=True), f'confusion_matrix_normalized_{state}.png')
    _save_figure(plot_per_class_metrics(metrics, class_names), f'per_class_metrics_{state}.png')

    return {
        'predictions': predictions,
        'confusion_matrix': cm,
        'metrics': metrics,
        'report': report,
        'misclassifications': misclass
    }


# ---------------------------------------------------------------------------
# Training & prediction
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    'tree_method': 'hist',
    'device': 'cpu',
    'max_bin': 128,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'nthread': -1,
    'seed': 42,
    'verbosity': 1
}
_STATE_PARAMS = {
    'SA': {'subsample': 1, 'colsample_bytree': 0.6, 'max_depth': 11, 'min_child_weight': 3, 'learning_rate': 0.1,
           'reg_lambda': 1, 'reg_alpha': 0.1, 'gamma': 0.4},
    'NSW': {'subsample': 0.8, 'colsample_bytree': 0.8, 'max_depth': 10, 'min_child_weight': 4, 'learning_rate': 0.5,
            'reg_lambda': 0.8, 'reg_alpha': 0.3, 'gamma': 0.4},
    'QLD': {'subsample': 0.8, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6, 'max_depth': 12, 'min_child_weight': 1,
            'learning_rate': 0.1, 'reg_lambda': 0.1, 'reg_alpha': 0, 'gamma': 0.4},
    'NT': {'subsample': 1, 'colsample_bytree': 0.6, 'max_depth': 11, 'min_child_weight': 3, 'learning_rate': 0.03,
           'reg_lambda': 1, 'reg_alpha': 0.1, 'gamma': 0.2},
    'TAS': {'subsample': 0.8, 'colsample_bytree': 0.6, 'max_depth': 12, 'min_child_weight': 4, 'learning_rate': 0.1,
            'reg_lambda': 1, 'reg_alpha': 0.55, 'gamma': 0.2},
    'VIC': {'subsample': 1, 'colsample_bytree': 0.8, 'max_depth': 15, 'min_child_weight': 4, 'learning_rate': 0.05,
            'reg_lambda': 10, 'reg_alpha': 0.1, 'gamma': 0.2},
    'WA': {'subsample': 0.8, 'colsample_bytree': 0.6, 'max_depth': 12, 'min_child_weight': 4, 'learning_rate': 0.1,
           'reg_lambda': 1, 'reg_alpha': 0.5, 'gamma': 0.2},
    'ACT': {'subsample': 1, 'colsample_bytree': 0.6, 'max_depth': 11, 'min_child_weight': 1, 'learning_rate': 0.03,
            'reg_lambda': 1, 'reg_alpha': 0.1, 'gamma': 0.4},
}
_STATE_NUM_ROUNDS = {'QLD': 2000, 'NSW': 3000, 'VIC': 2300, 'WA': 1800, 'SA': 1000, 'TAS': 1800, 'ACT': 1000, 'NT': 1000}


def ultra_fast_training_config(state):
    if state not in _STATE_PARAMS:
        raise ValueError(f"No training config for state '{state}'")
    return {**_BASE_PARAMS, **_STATE_PARAMS[state]}


def _load_model_frame(period, state):
    # df = pd.read_parquet(f"s3://PLACEHOLDER_PATH/{period}_all_res_training_{state}.parquet")
    df = pd.read_parquet(_training_output_path(period, state, pool="all"))
    df = df.drop(columns=NON_FEATURE_COLUMNS, errors='ignore')
    # XGBoost works in float32 internally, so this halves the frame without changing what the model sees
    float_columns = df.select_dtypes('float64').columns
    df[float_columns] = df[float_columns].astype('float32')
    return df


def _apply_training_categories(df, model):
    """Encode categoricals with the training categories so codes line up with the fitted trees."""
    categories = json.loads(model.attr('feature_categories'))
    for column, values in categories.items():
        df[column] = pd.Categorical(df[column], categories=values)
    return df


def train_test_model(period, state, method, collapse_other_classes=True):
    print("Starting Training and Testing of Model")
    start_time = time.perf_counter()
    df_training = _load_model_frame(period, state)
    df_training = df_training[
        df_training['DS_SOURCE'].str.contains(r'DCAN', case=False, na=False) & df_training['DWELLING_STRUCT'].notna()
    ].drop(columns='DS_SOURCE')
    labels = df_training.pop('DWELLING_STRUCT')
    if collapse_other_classes:
        # Fewer classes trains faster; pass collapse_other_classes=False to model each code separately
        labels = labels.mask(labels.isin(OTHER_DWELLING_CODES), 'OTHER')
    # Categories are set before the split so train and validation share the same encoding
    for column in CATEGORICAL_FEATURES:
        df_training[column] = df_training[column].astype("category")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    del labels

    X_train, X_val, y_train, y_val = train_test_split(
        df_training, y, test_size=0.1, random_state=42, stratify=y
    )
    del df_training
    params = ultra_fast_training_config(state)
    params['num_class'] = len(le.classes_)
    # QuantileDMatrix builds the histogram bins directly, avoiding a full float copy of the training data
    dtrain = xgb.QuantileDMatrix(X_train.drop(columns='ARID'), label=y_train, enable_categorical=True, max_bin=params['max_bin'])
    del X_train
    dval = xgb.QuantileDMatrix(X_val.drop(columns='ARID'), label=y_val, enable_categorical=True, max_bin=params['max_bin'], ref=dtrain)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=_STATE_NUM_ROUNDS[state],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    model.set_attr(feature_categories=json.dumps(
        {column: X_val[column].cat.categories.tolist() for column in CATEGORICAL_FEATURES}
    ))
    _log_time("", start_time)
    print("Completed Training and Testing of Model...")
    y_val = pd.DataFrame({'DWELLING_STRUCT': y_val}, index=X_val.index)
    return model, X_val, y_val, le, le.classes_.tolist()


def _probabilities_frame(probabilities, class_names, arids):
    probs_df = pd.DataFrame(probabilities, columns=class_names)
    probs_df['predicted_class'] = probs_df.idxmax(axis=1)
    probs_df['max_prob'] = probabilities.max(axis=1)
    probs_df['ARID'] = np.asarray(arids)
    return probs_df


def model_test_predictions(period, state, method, model_eval_output, X_val, y_val, class_names, le):
    probs_df = _probabilities_frame(model_eval_output['predictions'], class_names, X_val['ARID'])
    probs_df['true_class'] = le.inverse_transform(y_val['DWELLING_STRUCT'])
    # probs_df.to_csv(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_{state}_training_prediction_validation_output.csv", index=False)
    probs_df.to_csv(_output_path(f"{period}_{state}_training_prediction_validation_output.csv"), index=False)
    logging.info("Exported test predictions")


def predict_candidates(period, state, method, model, class_set, pool="all"):
    print("Starting Predicting Candidates...")
    df_cand = _load_model_frame(period, state)
    if pool == "all":
        df_cand = df_cand[df_cand['DS_SOURCE'].str.contains('IMPUTE', case=False, na=False) | df_cand['DWELLING_STRUCT'].isna()]
    elif pool == "no_ds":
        df_cand = df_cand[df_cand['DS_SOURCE'].isna() | (df_cand['DS_SOURCE'] == "")]
    elif pool == "dsam":
        df_cand = df_cand[df_cand['DS_SOURCE'].str.contains('ADMIN', case=False, na=False)]
    original_class = df_cand.pop('DWELLING_STRUCT')
    arids = df_cand.pop('ARID')
    df_cand = _apply_training_categories(df_cand.drop(columns='DS_SOURCE'), model)

    probs_df = _probabilities_frame(_predict_proba(model, df_cand), class_set, arids)
    probs_df['original_class'] = original_class.to_numpy()
    # probs_df.to_csv(f"s3://PLACEHOLDER_PATH/{period}_{state}_{pool}_predictions.csv", index=False)
    probs_df.to_csv(_output_path(f"{period}_{state}_{pool}_predictions.csv"), index=False)
    logging.info(f"Exported {period} {state} model for '{pool}' pool.")
    # The log file is now written straight to the output logs folder, so no upload/cleanup is needed:
    # s3_client = boto3.client('s3')
    # s3_client.upload_file(Filename=f"model_training_log_{state}.txt", Bucket="PLACEHOLDER", Key=f"PLACEHOLDER/model_training_log_{state}.txt")
    # with contextlib.suppress(FileNotFoundError):
    #     os.remove(f"model_training_log_{state}.txt")
    print(f"Completed predicting candidates for {state}.")
