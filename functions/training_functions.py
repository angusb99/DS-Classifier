import pandas as pd
import numpy as np
import time
import geopandas as gpd
from shapely import to_wkt
import sys
import math
import libpysal
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import logging
import contextlib
import os
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def start_log(state):
    logging.basicConfig(
        filename=f'model_training_log_{state}.txt',
        level=logging.INFO,
        format="%(message)s",
        filemode="w"
    )

# GENERATE ADR_CAD_LINKAGE_TRAINING - placeholders for all file paths
def generate_ar_cad_linkage(period, state, pool = "candidate"):
    start_time = time.perf_counter()
    print("Generating ADDR CAD Linkage")
    logging.info("Generating ADDR CAD Linkage: Loading files from s3")
    if pool == "training":
        ar_df = pd.read_parquet(f"s3://DATLAKE_PATH_PLACEHOLDER/{period}_dsim/{period}_training_{state}.parquet").groupby('ARID', as_index=False).first()
    elif pool == "candidate":
        ar_df = pd.read_parquet(f"s3://DATLAKE_PATH_PLACEHOLDER/{period}_dsim/{period}_candidate_{state}.parquet").groupby('ARID', as_index=False).first()
    else:
        ar_df = pd.read_parquet(f"s3://DATLAKE_PATH_PLACEHOLDER/{period}_dsim/{period}_all_res_{state}.parquet").groupby('ARID', as_index=False).first()
    cad_df = pd.read_parquet(f"s3://DATLAKE_PATH_PLACEHOLDER/{period}_cad/{period}_cad_{state}.parquet").drop(columns="MB_CODE_2021")
    logging.info("Generating ADDR Cad Linkage: Creating Geometries")
    cad_df['geometry'] = cad_df['CAD_POLY'].apply(wkt.loads)
    logging.info("Generating ADDR CAD Linkage: Creating geodataframes")
    cad_gdf = gpd.GeoDataFrame(cad_df, geometry="geometry", crs="EPSG:7844")
    cad_gdf = cad_gdf.to_crs("EPSG:9473")
    cad_gdf['CAD_PERIMETER'] = cad_gdf.geometry.length
    cad_gdf['CAD_COMPACTNESS'] = (
        4 * np.pi * cad_gdf['CAD_AREA'] /
        (cad_gdf['CAD_PERIMETER'] ** 2 + 1e-6)
    )
    cad_gdf = cad_gdf[cad_gdf['CAD_AREA'] > 100]
    cad_gdf = cad_gdf.to_crs('EPSG:7844')
    del cad_df
    ar_gdf = gpd.GeoDataFrame(ar_df, geometry=gpd.points_from_xy(ar_df.LONGITUDE, ar_df.LATITUDE), crs="EPSG:7844")
    del ar_df
    logging.info("Generating ADDR CAD Linkage: Computing join")
    joined = gpd.sjoin(ar_gdf, cad_gdf, how='left', predicate='intersects')
    missing_ar_gdf = ar_gdf[ar_gdf['ARID'].isin(joined[pd.isna(joined['CADASTRE_POLYGON_PID'])]['ARID'])]
    missing_ar_gdf = missing_ar_gdf.to_crs('EPSG:9473')
    cad_gdf = cad_gdf.to_crs('EPSG:9473')
    joined_nearest = missing_ar_gdf.sjoin_nearest(cad_gdf, how='inner', distance_col='dist')
    joined_nearest = joined_nearest.sort_values('dist').drop_duplicates(subset='ARID', keep='first')
    joined_nearest = joined_nearest.drop(columns=['geometry', 'index_right', 'CAD_POLY', 'dist'])
    del missing_ar_gdf, cad_gdf
    joined = joined[pd.notna(joined['CADASTRE_POLYGON_PID'])]
    joined = joined.drop(columns=['geometry', 'index_right', 'CAD_POLY'])
    joined = pd.concat([joined, joined_nearest])
    # Zipped coords for comparison
    ar_cad = joined
    ar_cad['coord'] = list(zip(ar_cad['LATITUDE', ar_cad['LONGITUDE']]))
    ar_cad_counts = ar_cad.groupby('CADASTRE_POLYGON_PID').agg(
        row_count=('coord', 'size'),
        dist_coord_count=('coord', lambda s: pd.Series(s).dropna().nunique())
    ).reset_index()
    ar_mhe_for_review = ar_cad[ar_cad['CADASTRE_POLYGON_PID'].isin(
        ar_cad_counts[(ar_cad_counts['row_count'] == ar_cad_counts['dist_coord_count']) &
                      (ar_cad_counts['dist_coord_count'] > 6)]['CADASTRE_POLYGON_PID']
    )][['ARID', 'CADASTRE_POLYGON_PID']]
    end_time = time.perf_counter()
    print(f"Execution Time: {(end_time - start_time):.4f} seconds")
    logging.info(f"Execution Time: {(end_time - start_time):.4f} seconds")
    return joined, ar_mhe_for_review

def generate_bld_cad_linkage(period, state, extended_features=False):
    start_time = time.perf_counter()
    logging.info("Loading files from s3")
    bld_df = pd.read_parquet(f's3://PLACEHOLDER_FILE_PATH/{period}_bld/{period}_bld_{state}.parquet').drop(columns='MB_CODE_2021')
    cad_df = pd.read_parquet(f's3://PLACEHOLDER_FILE_PATH/{period}_cad/{period}_cad_{state}.parquet').drop(columns=['MB_CODE_2021', 'CAD_POLY_PID_COUNT'])
    logging.info("Creating geometries...")
    cad_df['geometry'] = cad_df['CAD_POLY'].apply(wkt.loads)
    bld_df['geometry'] = bld_df['BLD_POLY'].apply(wkt.loads)
    logging.info("Creating gdfs")
    cad_gdf = gpd.GeoDataFrame(cad_df, geometry="geometry", crs="EPSG:7844")
    del cad_df
    bld_gdf = gpd.GeoDataFrame(bld_df, geometry="geometry", crs="EPSG:7844")
    del bld_df
    if extended_features = True:
        bld_gdf = bld_gdf.to_crs("EPSG:3577")
        bld_gdf_left = bld_gdf.copy()
        bld_gdf_left['_temp_id'] = range(len(bld_gdf_left))
        nearest = gpd.sjoin_nearest(
            bld_gdf_left,
            bld_gdf,
            distance_col = 'distance',
            how='left'
        )
        nearest = nearest[nearest['_temp_id'] != nearest.index_right]