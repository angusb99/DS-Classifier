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
        def calc_shared_boundary_length(geom_left, geom_right):
            if geom_left is None or geom_right is None:
                return 0.0
            inter = geom_left.boundar.intersection(geom_right.boundary)
            return float(getattr(inter, "length", 0.0))
        nearest['BLD_SHARED_LENGTH'] = [
            calc_shared_boundary_length(gl, gr)
            for gl, gr in zip(nearest["geometry"], bld_gdf.loc[nearest["index_right"], "geometry"])
        ]
        max_shared = nearest.groupby('_temp_id')['BLD_SHARED_LENGTH'].max().reset_index()
        bld_gdf['BLD_MAX_SHARED_LENGTH'] = max_shared['BLD_SHARED_LENGTH']
        min_distances = nearest.groupby('_temp_id')['distance'].min().reset_index()
        del bld_gdf_left
        bld_gdf['BLD_NEAREST_DISTANCE'] = min_distances['distance']
        bld_gdf['BLD_TOUCH'] = np.where(bld_gdf['BLD_NEAREST_DISTANCE'] < 0.001, 1, 0)
        del nearest
        bld_gdf  = bld_gdf.to_crs('EPSG:7844')
        W = libpysal.weights.Queen.from_dataframe(bld_gdf, use_index=False, silence_warnings=True)
        components = W.component_labels
        bld_gdf = bld_gdf.dissolve(by = components, aggfunc={
            'BUILDING_AREA': 'sum',
            'QUALITY_CLASS': 'first',
            'BLD_MAX_SHARED_LENGTH': 'max',
            'BLD_TOUCH': 'sum',
            'BLD_NEAREST_DISTANCE': 'min',
            'BUILDING_NUM_VERTICES': 'sum',
            'ADDRESS_COUNT': 'max'
        })
    logging.info('Computing join...')
    joined = gpd.sjoin(bld_gdf, cad_gdf, how='left', predicate='intersects')
    valid_mask = joined['index_right'].notna()
    joined_valid = joined[valid_mask].copy()
    logging.info("Reprojecting for real world area...")
    cad_gdf = cad_gdf.to_crs('EPSG:3577')
    bld_gdf = bld_gdf.to_crs('EPSG:3577')
    logging.info('Obtain geometries and compute area overlap...')
    bld_geoms = bld_gdf.loc[joined_valid.index, 'geometry']
    cad_geoms = cad_gdf.loc[joined_valid['index_right'], 'geometry']
    joined_valid['BLD_PERIMETER'] = [b.length for b in bld geoms]
    joined_valid['BLD_INTERSECTION_AREA'] = [b.intersection(p).area for b, p in zip(bld_geoms, cad_geoms)]
    joined_valid['PCT_OF_BLD_INTERSECT'] = (joined_valid['BLD_INTERSECTION_AREA'] / joined_valid['BUILDING_AREA'])*100
    joined_valid['BLD_CROSS_PARCEL'] = np.where(joined_valid['PCT_OF_BLD_INTERSECT'] < 100, 1, 0)
    joined_valid['BLD_CAD_DISTANCE'] = [b.distance(p.boundary) for b, p in zip(bld_geoms, cad_geoms)]
    joined_valid['BLD_CAD_BOUNDARY_INTERSECTION'] = [b.intersection(p.boundary).length for b, p in zip(bld_geoms, cad_geoms)]
    joined_valid = joined_valid[joined_valid['PCT_OF_BLD_INTERSECT'] > 10]
    joined_valid = joined_valid.drop(columns=['geometry', 'index_right', 'CAD_POLY', 'CADASTRE_PID'])'])
    if extended_features == True:
        building_agg = joined_valid.groupby('CADASTRE_POLYGON_PID').agg({
            'BUILDING_AREA': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'BLD_PERIMETER': ['mean', 'sum'],
            'BLD_INTERSECTION_AREA': ['min', 'mean', 'max', 'sum'],
            'PCT_OF_BLD_INTERSECT': ['min', 'mean', 'max'],
            'ADDRESS_COUNT': 'sum',
            'QUALITY_CLASS': 'first',
            'BLD_NEAREST_DISTANCE': 'min',
            'BLD_MAX_SHARED_LENGTH': ['min', 'mean', 'max', 'sum'],
            'BLD_TOUCH': 'sum',
            'BLD_CAD_DISTANCE': ['min', 'mean', 'max', 'sum'],
            'BLD_CAD_BOUNDARY_INTERSECTION': ['min', 'mean', 'max', 'sum'],
            'BLD_CROSS_PARCEL': 'sum'
        }).reset_index()
        # Flatten column names
        building_agg.columns = [
            'CADASTRE_POLYGON_PID',
            'NUM_BUILDINGS',
            'TOTAL_BUILDING_AREA',
            'MEAN_BUILDING_AREA',
            'STD_BUILDING_AREA',
            'MIN_BUILDING_AREA',
            'MAX_BUILDING_AREA',
            'MEAN_BUILDING_PERIMETER',
            'TOTAL_BUILDING_PERIMETER',
            'MIN_BUILDING_INTERSECTION_AREA',
            'MEAN_BUILDING_INTERSECTION_AREA',
            'MAX_BUILDING_INTERSECTION_AREA',
            'TOTAL_BUILDING_INTERSECTION_AREA',
            'MIN_PCT_BUILDING_INTERSECTION',
            'MEAN_PCT_BUILDING_INTERSECTION',
            'MAX_PCT_BUILDING_INTERSECTION',
            'TOTAL_BLD_ADDRESS_COUNT',
            'BLD_QUALITY_CLASS',
            'MIN_BLD_NEAREST_DISTANCE',
            'MIN_BLD_MAX_SHARED_LENGTH',
            'MEAN_BLD_MAX_SHARED_LENGTH',
            'SUM_BLD_MAX_SHARED_LENGTH',
            'SUM_BLD_TOUCHING',
            'MIN_BLD_CAD_DISTANCE',
            'MEAN_BLD_CAD_DISTANCE',
            'MAX_BLD_CAD_DISTANCE',
            'SUM_BLD_CAD_DISTANCE',
            'MIN_BLD_CAD_BOUNDARY_INTERSECTION',
            'MEAN_BLD_CAD_BOUNDARY_INTERSECTION',
            'MAX_BLD_CAD_BOUNDARY_INTERSECTION',
            'SUM_BLD_CAD_BOUNDARY_INTERSECTION',
            'SUM_BLD_CROSS_PARCEL'
        ]
    else:
        building_agg = joined_valid.groupby('CADASTRE_POLYGON_PID').agg({
            'BUILDING_AREA': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'BLD_PERIMETER': ['mean', 'sum'],
            'BLD_INTERSECTION_AREA': ['min', 'mean', 'max', 'sum'],
            'PCT_OF_BLD_INTERSECT': ['min', 'mean', 'max'],
            'ADDRESS_COUNT': 'sum',
            'QUALITY_CLASS': 'first',
            'BLD_CAD_DISTANCE': ['min', 'mean', 'max', 'sum'],
            'BLD_CAD_BOUNDARY_INTERSECTION': ['min', 'mean', 'max', 'sum'],
            'BLD_CROSS_PARCEL': 'sum'
        }).reset_index()
        building_agg.columns = [
            'CADASTRE_POLYGON_PID',
            'NUM_BUILDINGS',
            'TOTAL_BUILDING_AREA',
            'MEAN_BUILDING_AREA',
            'STD_BUILDING_AREA',
            'MIN_BUILDING_AREA',
            'MAX_BUILDING_AREA',
            'MEAN_BUILDING_PERIMETER',
            'TOTAL_BUILDING_PERIMETER',
            'MIN_BUILDING_INTERSECTION_AREA',
            'MEAN_BUILDING_INTERSECTION_AREA',
            'MAX_BUILDING_INTERSECTION_AREA',
            'TOTAL_BUILDING_INTERSECTION_AREA',
            'MIN_PCT_BUILDING_INTERSECTION',
            'MEAN_PCT_BUILDING_INTERSECTION',
            'MAX_PCT_BUILDING_INTERSECTION',
            'TOTAL_BLD_ADDRESS_COUNT',
            'BLD_QUALITY_CLASS',
            'MIN_BLD_CAD_DISTANCE',
            'MEAN_BLD_CAD_DISTANCE',
            'MAX_BLD_CAD_DISTANCE',
            'SUM_BLD_CAD_DISTANCE',
            'MIN_BLD_CAD_BOUNDARY_INTERSECTION',
            'MEAN_BLD_CAD_BOUNDARY_INTERSECTION',
            'MAX_BLD_CAD_BOUNDARY_INTERSECTION',
            'SUM_BLD_CAD_BOUNDARY_INTERSECTION',
            'SUM_BLD_CROSS_PARCEL'
        ]
    
    building_agg['BUILDING_SIZE_CV'] = (
        building_agg['STD_BUILDING_AREA'] /
        (building_agg['MEAN_BUILDING_AREA'] + 1e-6)
    )

    building_agg['BLD_SIZE_RANGE_RATIO'] = (
        building_agg['MAX_BUILDING_AREA'] /
        (building_agg['MIN_BUILDING_AREA'] + 1e-6)
    )
    end_time = time.perf_counter()
    print(f"Execution Time: {(end_time - start_time):.4f} seconds")
    return building_agg

def generate_ar_building_attachment(ar_cad, period, state, pool):
    # Create set of zipped coords for comparison
    ar_cad['coord'] = list(zip(ar_cad['LATITUDE'], ar_cad['LONGITUDE']))
    ar_cad_counts = ar_cad.groupby('CADASTRE_POLYGON_PID').agg(
        row_count=('coord', 'size'),
        dist_coord_count=('coord', lambda s: pd.Series(s).dropna().nunique())
    ).reset_index()
    # Where there is the same number of residential ADDR and distinct geocodes on a parcel, these are direct candidates for joining to specific buildings
    direct_building_join_cand = ar_cad[ar_cad['CADASTRE_POLYGON_PID'].isin(
        ar_cad_counts[(ar_cad_counts['row_count'] == ar_cad_counts['dist_coord_count']) &
        (ar_cad_counts['dist_coord_count'] > 1)]['CADASTRE_POLYGON_PID']
    )]
    bld_df = pd.read_parquet(f's3://PLACEHOLDER_PATH/{period}_bld/{period}_bld_{state}.parquet').drop(columns='MB_CODE_2021')
    bld_df['geometry'] = bld_df['BLD_POLY'].apply(wkt.loads)
    bld_gdf = gpd.GeoDataFrame(bld_df, geometry="geometry", crs="EPSG:7844")
    bld_gdf = bld_gdf.to_crs("EPSG:7844")
    W = libpysal.weights.Queen.from_dataframe(bld_gdf, use_index=False, silence_warnings=True)
    components = W.component_labels
    bld_gdf = bld_gdf.dissolve(by=components, aggfunc={
        'BUILDING_PID': 'first',
        'BUILDING_AREA': 'sum',
        'QUALITY_CLASS': 'first',
        'BUILDING_NUM_VERTICES': 'sum',
        'ADDRESS_COUNT': 'max'
    })
    bld_gdf['NEW_ID'] = range(len(bld_gdf))
    bld_gdf = bld_gdf.to_crs('EPSG:3577')
    ar_gdf = gpd.GeoDataFrame(direct_building_join_cand, geometry=gpd.points_from_xy(direct_building_join_cand.LONGITUDE, direct_building_join_cand.LATITUDE), crs="EPSG:7844")
    ar_gdf = ar_gdf.to_crs('EPSG:3577')
    nearest = gpd.sjoin_nearest(
        ar_gdf,
        bld_gdf,
        distance_col = 'distance',
        how='left'
    )
    acceptable_distance = nearest[nearest['distance'] < 5]
    regular_treatment_pool = ar_cad[~ar_cad['ARID'].isin(acceptable_distance['ARID'])]
    cad_df = pd.read_parquet(f's3://PLACEHOLDER_PATH/{period}_cad/{period}_cad_{state}.parquet').drop(columns=['MB_CODE_2021', 'CAD_POLY_PID_COUNT'])
    cad_df['geometry'] = cad_df['CAD_POLY'].apply(wkt.loads)
    cad_gdf = gpd.GeoDataFrame(cad_df, geometry="geometry", crs="EPSG:7844")
    bld_gdf = bld_gdf.to_crs("EPSG:7844")
    joined = gpd.sjoin(bld_gdf, cad_gdf, how='left', predicate='intersects')
    valid_mask = joined['index_right'].notna()
    joined_valid = joined[valid_mask].copy()
    logging.info("Reprojecting for real world area...")
    cad_gdf = cad_gdf.to_crs('EPSG:3577')
    bld_gdf = bld_gdf.to_crs('EPSG:3577')
    logging.info("Obtain geometries and compute area ovrlap...")
    bld_geoms = bld_gdf.loc[joined_valid.index, 'geometry']
    cad_geoms = cad_gdf.loc[joined_valid['index_right'], 'geometry']
    joined_valid['BLD_PERIMETER'] = [b.length for b in bld_geoms]
    joined_valid['BLD_INTERSECTION_AREA'] = [b.intersection(p).area for b, p in zip(bld_geoms, cad_geoms)]
    joined_valid['PCT_OF_BLD_INTERSECT'] = (joined_valid['BLD_INTERSECTION_AREA'] / joined_valid['BUILDING_AREA'])*100
    joined_valid['BLD_CROSS_PARCEL'] = np.where(joined_valid['PCT_OF_BLD_INTERSECT'] < 100, 1, 0)
    joined_valid['BLD_CAD_DISTANCE'] = [b.distance(p.boundary) for b, p in zip(bld_geoms, cad_geoms)]
    joined_valid['BLD_CAD_BOUNDARY_INTERSECTION'] = [b.intersection(p.boundary).length for b, p in zip(bld_geoms, cad_geoms)]
    joined_valid = joined_valid[(joined_valid['PCT_OF_BLD_INTERSECT'] > 20) | (joined_valid['PCT_OF_BLD_INTERSECT'] == 0)]
    joined_valid = joined_valid.drop(columns=['geometry', 'index_right', 'CAD_POLY', 'CADASTRE_PID'])
    joined_valid = joined_valid[joined_valid['CADASTRE_POLYGON_PID'].isin(acceptable_distance['CADASTRE_POLYGON_PID']) | (pd.isna(joined_valid['CADASTRE_POLYGON_PID']))]
    building_agg = joined_valid.groupby('CADASTRE_POLYGON_PID').agg({
            'BUILDING_AREA': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'BLD_PERIMETER': ['mean', 'sum'],
            'BLD_INTERSECTION_AREA': ['min', 'mean', 'max', 'sum'],
            'PCT_OF_BLD_INTERSECT': ['min', 'mean', 'max'],
            'ADDRESS_COUNT': 'sum',
            'QUALITY_CLASS': 'first',
            'BLD_CAD_DISTANCE': ['min', 'mean', 'max', 'sum'],
            'BLD_CAD_BOUNDARY_INTERSECTION': ['min', 'mean', 'max', 'sum'],
            'BLD_CROSS_PARCEL': 'sum'
    }).reset_index()
    building_agg.columns = [
            'CADASTRE_POLYGON_PID',
            'NUM_BUILDINGS',
            'TOTAL_BUILDING_AREA',
            'MEAN_BUILDING_AREA',
            'STD_BUILDING_AREA',
            'MIN_BUILDING_AREA',
            'MAX_BUILDING_AREA',
            'MEAN_BUILDING_PERIMETER',
            'TOTAL_BUILDING_PERIMETER',
            'MIN_BUILDING_INTERSECTION_AREA',
            'MEAN_BUILDING_INTERSECTION_AREA',
            'MAX_BUILDING_INTERSECTION_AREA',
            'TOTAL_BUILDING_INTERSECTION_AREA',
            'MIN_PCT_BUILDING_INTERSECTION',
            'MEAN_PCT_BUILDING_INTERSECTION',
            'MAX_PCT_BUILDING_INTERSECTION',
            'TOTAL_BLD_ADDRESS_COUNT',
            'BLD_QUALITY_CLASS',
            'MIN_BLD_CAD_DISTANCE',
            'MEAN_BLD_CAD_DISTANCE',
            'MAX_BLD_CAD_DISTANCE',
            'SUM_BLD_CAD_DISTANCE',
            'MIN_BLD_CAD_BOUNDARY_INTERSECTION',
            'MEAN_BLD_CAD_BOUNDARY_INTERSECTION',
            'MAX_BLD_CAD_BOUNDARY_INTERSECTION',
            'SUM_BLD_CAD_BOUNDARY_INTERSECTION',
            'SUM_BLD_CROSS_PARCEL'
        ]
    building_agg['BUILDING_SIZE_CV'] = (
        building_agg['STD_BUILDING_AREA'] /
        (building_agg['MEAN_BUILDING_AREA'] + 1e-6)
    )

    building_agg['BLD_SIZE_RANGE_RATIO'] = (
        building_agg['MAX_BUILDING_AREA'] /
        (building_agg['MIN_BUILDING_AREA'] + 1e-6)
    )
    joined_df = acceptable_distance.merge(building_agg, how='left', on='NEW_ID')
    joined_df = joined_df.drop_duplicates(subset='ARID')
    joined_df['ARIDS_PER_PARCEL'] = joined_df.groupby('NEW_ID')['ARID'].transform('nunique')
    joined_df['ARIDS_PER_BUILDING_RATIO'] = joined_df['ARIDS_PER_PARCEL'] / joined_df['NUM_BUILDINGS']
    joined_df['BUILDING_AREA_PER_ARID'] = (
        joined_df['TOTAL_BUILDING_AREA'] / joined_df['ARIDS_PER_PARCEL']
    )
    joined_df['CAD_AREA_PER_ARID'] = (
        joined_df['CAD_AREA'] / joined_df['ARIDS_PER_PARCEL']
    )
    joined_df['ARID_DENSITY'] = (
        joined_df['ARIDS_PER_PARCEL'] / joined_df['CAD_AREA']
    )
    joined_df['BUILDING_COVERAGE_RATIO'] = (
        joined_df['TOTAL_BUILDING_AREA'] / joined_df['CAD_AREA']
    )
    joined_df['LOG_CAD_AREA'] = np.log1p(joined_df['CAD_AREA'])
    joined_df['LOG_BUILDING_TOTAL_AREA'] = np.log1p(joined_df['TOTAL_BUILDING_AREA'])
    joined_df['LOG_ARID_PER_PARCEL'] = np.log1p(joined_df['ARIDS_PER_PARCEL'])
    joined_df['BUILDING_TO_CAD_SIZE_RATIO'] = (
        joined_df['MEAN_BUILDING_AREA'] / joined_df['CAD_AREA']
    )
    joined_df = joined_df.drop(columns = ['coord', 'geometry', 'index_right', 'BUILDING_PID',
    'BUILDING_AREA', 'QUALITY_CLASS', 'BUILDING_NUM_VERTICES', 'ADDRESS_COUNT', 'NEW_ID', 'distance'])
    return joined_df

def generate_training_data_workflow(period, state, pool, method, extended_features):
    init_time = time.perf_counter()
    ar_cad, ar_mhe_for_review = generate_ar_cad_linkage(period, state, pool)
    ar_cad.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}/{state}/{state}_ar_cad_linkage.parquet", index=False)
    bld_cad = generate_bld_cad_linkage(period, state, extended_features)
    joined_df = ar_cad.merge(bld_cad, how='left', on='CADASTRE_POLYGON_PID')
    joined_df = joined_df.drop_duplicates(subset='ARID')
    joined_df['ARIDS_PER_PARCEL'] = joined_df.groupby('CADASTRE_POLYGON_PID')['ARID'].transform('nunique')
    joined_df['ARIDS_PER_BUILDING_RATIO'] = joined_df['ARIDS_PER_PARCEL'] / joined_df['NUM_BUILDINGS']
    joined_df['BUILDING_AREA_PER_ARID'] = (
        joined_df['TOTAL_BUILDING_AREA'] / joined_df['ARIDS_PER_PARCEL']
    )
    joined_df['CAD_AREA_PER_ARID'] = (
        joined_df['CAD_AREA'] / joined_df['ARIDS_PER_PARCEL']
    )
    joined_df['ARID_DENSITY'] = (
        joined_df['ARIDS_PER_PARCEL'] / joined_df['CAD_AREA']
    )
    joined_df['BUILDING_COVERAGE_RATIO'] = (
        joined_df['TOTAL_BUILDING_AREA'] / joined_df['CAD_AREA']
    )
    joined_df['LOG_CAD_AREA'] = np.log1p(joined_df['CAD_AREA'])
    joined_df['LOG_BUILDING_TOTAL_AREA'] = np.log1p(joined_df['TOTAL_BUILDING_AREA'])
    joined_df['LOG_ARID_PER_PARCEL'] = np.log1p(joined_df['ARIDS_PER_PARCEL'])
    joined_df['BUILDING_TO_CAD_SIZE_RATIO'] = (
        joined_df['MEAN_BUILDING_AREA'] / joined_df['CAD_AREA']
    )
    if method == "building_target":
        ar_bld = generate_ar_building_attachment(ar_cad=ar_cad, period=period, state=state, pool=pool)
        if extended_features == True:
            joined_subset = joined_df[~joined_df['ARID'].isin(ar_bld['ARID'])].drop(columns=['MIN_BLD_NEAREST_DISTANCE', 'MIN_BLD_MAX_SHARED_LENGTH',
            'MAX_BLD_MAX_SHARED_LENGTH', 'SUM_BLD_MAX_SHARED_LENGTH', 'SUM_BLD_TOUCHING'])
        else:
            joined_subset = joined_df[~joined_df['ARID'].isin(ar_bld['ARID'])]
        bld_target_joined = pd.concat([joined_subset, ar_bld], axis=0)
        if pool == "training":
            bld_target_joined.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{period}_full_training_{state}.parquet", index=False)
        elif pool == "candidate":
            bld_target_joined.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{period}_full_candidate_{state}.parquet", index=False)
        else:
            bld_target_joined.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{period}_all_res_training{state}.parquet", index=False)
    else:
        if pool == "training":
            joined_df.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{period}_full_training_{state}.parquet", index=False)
        elif pool == "candidate":
            joined_df.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{period}_full_candidate_{state}.parquet", index=False)
        else:
            joined_df.to_parquet(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_DSIM/{state}/{period}_all_res_training_{state}.parquet", index=False)
    ar_mhe_for_review.to_csv(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}/{state}/{period}_{state}_mhe_for_review.csv", index=False)
    current_time = time.perf_counter()
    logging.info(f"Completed {state}. Execution Time: {(current_time - init_time):.4f} seconds")
    print(f"Completed {state}. Execution Time: {(current_time - init_time):.4f} seconds")


def get_confusion_matrix(model, X_test, y_test):
    # Make predictions
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    predictions = model.predict(dtest)
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    return cm, predictions

def plot_confusion_matrix(cm, class_names=None, normalize=False, figsize=(10,8)):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'
    # Create labels
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws = {'label':'Count' if not normalize else 'Proportion'}
    )

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    return plt

def calculate_all_metrics(y_test, predictions, class_names=None)
    # Basic
    accuracy = accuracy_score(y_test, predictions)
    
    # Multi class
    precision_macro = precision_score(y_test, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, predictions, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)

    # Per class
    precision_per_class = precision_score(y_test, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, predictions, average=None, zero_division=0)
    # Additional metrics
    kappa = cohen_kappa_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc
    }
    return metrics

def print_metrics_summary(metrics, class_names=None):
    logging.info("\n" + "=" * 70)
    logging.info("MODEL PERFORMANCE METRICS - TEST")
    logging.info("="*70)
    logging.info("OVERALL METICS")
    logging.info("="*70)
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
    logging.info("-"*70)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['precision_per_class']))]
    logging.info(f"{'Class':<20} {'Precision': <12} {'Recall': <12} {'F1-Score':<12}")
    logging.info("-"*70)

    for i, name in enumerate(class_names):
        prec = metrics['precision_per_class'][i]
        rec = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        logging.info(f"{name:<20} {prec:>11.4f} {rec:>11.4f} {f1:>11.4f}")
    
    logging.info("="*70)


def get_classification_report(y_test, predictions, class_names=None):
    if class_names is not None:
        target_names = class_names
    else:
        target_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    report = classification_report(
        y_test,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    return report

def analyze_per_class_performance(cm, class_names = None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    logging.info("\n" + "="*70)
    logging.info("PER-CLASS DETAILED ANALYSIS")
    logging.info("="*70)
    for i, name in enumerate(class_names):
        logging.info(f"\n{name}:")
        logging.info("-"*70)
        tp = cm[i,i]
        fp = cm[:, i].sum() -tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_actual = cm[i, :].sum()
        total_predicted = cm[:, i].sum()
        logging.info(f" True Positives:     {tp:>6g}")
        logging.info(f" False Positives:    {fp:>6g}")
        logging.info(f" False Negatives:    {fn:>6g}")
        logging.info(f" True Negatives:     {tn:>6g}")
        logging.info(f" Total Actual:       {total_actual:>6g}")
        logging.info(f" True Predicted:     {total_predicted:>6g}")
        logging.info(f" Precision:          {precision:>6.4f}")
        logging.info(f" Recall:             {recall:>6.4f}")
        logging.info(f" F1-Score:           {f1:>6g}")


def plot_per_class_metrics(metrics, class_names=None, figsize=(12,6)):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['precision_per_class']))]

    # Prep data
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=figsize)

    # Bars
    bars1 = ax.bar(x - width, metrics['preicsion_per_class'], width, label = 'Precision', alpha=0.8)
    bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', alpha=0.8)

    # Customise
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_titel('Per-Class Performance Metrics', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    return plt

def misclassification_analysis(y_test, predictions, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    cm = confusion_matrix(y_test, predictions)
    logging.info("\n" + "="*70)
    logging.info("MISCLASSIFICATION ANALYSIS")
    logging.info("="*70)

    # Most common misclassifications
    misclass = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                misclass.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': cm[i, j]
                    'true_idx': i,
                    'pred_idx': j
                })

    # Sort by count
    misclass.sort(key=lambda x: x['count'], reverse=True)
    logging.info('\n Top 10 Misclassification Patterns:')
    logging.info('-'*70)
    logging.info(f"{'True Class':<20} -> {'Predicted': <20} {'Count':>10}")
    logging.info('-'*70)

    for item in misclass[:10]:
        print(f"{item['true']:<20} -> {item['predicted']:<20} {item['count']:>10}")

    return misclass


def complete_model_evaluation(period, state, method, model, X_test, y_test, class_names=None):
    logging.info("\n" + "="*70)
    logging.info("COMPLETE MODEL EVALUATION")
    logging.info("="*70)

    # Step 1: Get Predictions
    logging.info("\n[1] Making predictions on test set...")
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    predictions = model.predict(dtest)
    class_predictions = predictions.argmax(axis=1)

    # 2: Confusion Matrix
    cm = confusion_matrix(y_test, class_predictions)

    # 3: Metrics
    metrics = calculate_all_metrics(y_test, class_predictions, class_names)
    print_metrics_summary(metrics, class_names)

    # 4: Classification report
    report = get_classification_report(y_test, class_predictions, class_names)
    logging.info("\n" + report)

    # 5: Per-Class Analysis
    analyze_per_class_performance(cm, class_names)

    # 6: Misclassification Analysis
    misclass = misclassification_analysis(y_test, class_predictions, class_names)

    # 7: Viz
    plt1 = plot_confusion_matrix(cm, class_names, normalize=False)
    plt1.savefig(f'confusion_matrix_counts_{state}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt2 = plot_confusion_matrix(cm, class_names, normalize=True)
    plt2.savefig(f'confusion_matrix_normalized_{state}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt3 = plot_per_class_metrics(metrics, class_names)
    plt3.savefig(f'per_class_metrics_{state}.png', dpi=300, bbox_inches='tight')
    plt.close()

    s3_client = boto3.client('s3')
    s3_client.upload_file(Filename=f"confusion_matrix_counts_{state}.png", Bucket="PLACEHOLDER", Key="PLACEHOLDER")
    s3_client.upload_file(Filename=f"confusion_matrix_normalized_{state}.png", Bucket="PLACEHOLDER", Key="PLACEHOLDER")
    s3_client.upload_file(Filename=f"per_class_metrics_{state}.png", Bucket="PLACEHOLDER", Key="PLACEHOLDER")
    with contextlib.supress(FileNotFoundError):
        os.remove(f"confusion_matrix_counts_{state}.png")
        os.remove(f"confusion_matrix_normalized_{state}.png")
        os.remove(f"per_class_metrics_{state}.png")
    
    results = {
        'predictions': predictions,
        'confusion_matrix': cm,
        'metrics': metrics,
        'report': report,
        'misclassifications': misclass
    }

    return results

def ultra_fast_training_config(state):
    if state == "SA":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 1,
            'colsample_bytree': 0.6,
            'max_depth': 11,
            'min_child_weight': 3,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 1,
            'reg_alpha': 0.1,
            'gamma': 0.4,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "NSW":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': 10,
            'min_child_weight': 4,
            'learning_rate': 0.5,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 0.8,
            'reg_alpha': 0.3,
            'gamma': 0.4,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "QLD":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'colsample_bylevel': 0.6,
            'max_depth': 12,
            'min_child_weight': 1,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 0.1,
            'reg_alpha': 0,
            'gamma': 0.4,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "NT":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 1,
            'colsample_bytree': 0.6,
            'max_depth': 11,
            'min_child_weight': 3,
            'learning_rate': 0.03,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 1,
            'reg_alpha': 0.1,
            'gamma': 0.2,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "TAS":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'max_depth': 12,
            'min_child_weight': 4,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 1,
            'reg_alpha': 0.55,
            'gamma': 0.2,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "VIC":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 1,
            'colsample_bytree': 0.8,
            'max_depth': 15,
            'min_child_weight': 4,
            'learning_rate': 0.05,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 10,
            'reg_alpha': 0.1,
            'gamma': 0.2,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "WA":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'max_depth': 12,
            'min_child_weight': 4,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 1,
            'reg_alpha': 0.5,
            'gamma': 0.2,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    elif state == "ACT":
        params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 128,
            'subsample': 1,
            'colsample_bytree': 0.6,
            'max_depth': 11,
            'min_child_weight': 1,
            'learning_rate': 0.03,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'reg_lambda': 1,
            'reg_alpha': 0.1,
            'gamma': 0.4,
            'n_jobs': -1,
            'seed': 42,
            'verbosity': 1
        }
    return params


def train_test_model(period, state, method):
    print("Starting Training and Testing of Model")
    start_time = time.perf_counter()
    df_training = pd.read_parquet(f"s3://PLACEHOLDER_PATH/{period}_all_res_training_{state}.parquet").drop(columns=['STATE_ABBREVIATION', "CADASTRE_POLYGON_PID", 'CADASTRE_PID', 'coord'])
    df_training = df_training[df_training['DS_SOURCE'].str.contains(r'DCAN', case=False, na=False) & pd.notna(df_training['DWELLING_STRUCT'])].drop(columns='DS_SOURCE')
    df_training['DWELLING_STRUCT'] = np.where(
        df_training['DWELLING_STRUCT'].isin(['x35', 'x91', 'x92', 'x93', 'x94']), 'OTHER',
        df_training['DWELLING_STRUCT']
    ) # Potentially remove this classification to speed it up a bit
    labelled_result = df_training[['ARID', 'DWELLING_STRUCT']]
    # Complete workflow
    X_train, X_val, y_train, y_val = train_test_split(
        df_training.drop(columns=['DWELLING_STRUCT']),
        df_training[['DWELLING_STRUCT']],
        test_size=0.1, random_state=42, stratify=df_training[['DWELLING_STRUCT']]
    )
    X_train['MESHBLOCK'] = X_train['MESHBLOCK'].astype("category")
    X_train['BLD_QUALITY_CLASS'] = X_train['BLD_QUALITY_CLASS'].astype("category")
    le = LabelEncoder()
    y_train['DWELLING_STRUCT'] = y_train['DWELLING_STRUCT'].astype("category")
    y_train['DWELLING_STRUCT'] = le.fit_transform(y_train['DWELLING_STRUCT'])
    dtrain = xgb.DMatrix(X_train.drop(columns='ARID'), label=y_train, enable_categorical=True)
    X_val['MESHBLOCK'] = X_val['MESHBLOCK'].astype("category")
    X_val['BLD_QUALITY_CLASS'] = X_val['BLD_QUALITY_CLASS'].astype("category")
    class_set = y_val['DWELLING_STRUCT'].drop_duplicates().sort_values().tolist()
    y_val['DWELLING_STRUCT'] = y_val['DWELLING_STRUCT'].astype("category")
    y_val['DWELLING_STRUCT'] = le.fit_transform(y_val['DWELLING_STRUCT'])
    dval = xgb.DMatrix(X_val.drop(columns='ARID'), label=y_val, enable_categorical=True)
    params = ultra_fast_training_config(state)
    params['num_class'] = len(np.unique(df_training['DWELLING_STRUCT']))
    if state == "QLD":
        num_rounds = 2000
    elif state == "NSW":
        num_rounds = 3000
    elif state == "VIC":
        num_rounds = 2300
    elif state == "WA":
        num_rounds = 1800
    elif state == "SA":
        num_rounds = 1000
    elif state == "TAS":
        num_rounds = 1800
    elif state == "ACT":
        num_rounds = 1000
    elif state == "NT":
        num_rounds = 1000
    else:
        num_rounds = 300
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    end_time = time.perf_counter()
    logging.info(f"Execution Time: {(end_time - start_time):.4f} seconds")
    print(f"Execution Time: {(end_time - start_time):.4f} seconds")
    print("Completed Training and Testing of Model...")
    return model, X_val, y_val, le, class_set


def model_test_predictions(period, state, method, model_eval_output, X_val, y_val, class_names, le):
    probs_df = pd.DataFrame(model_eval_output['predictions'], columns=class_names)
    probs_df['predicted_class'] = probs_df.idxmax(axis=1)
    probs_df['max_prob'] = probs_df[class_names].max(axis=1)
    probs_df['ARID'] = X_val.iloc[:,0].values
    probs_df['true_class'] = le.inverse_transform(y_val['DWELLING_STRUCTURE'])
    probs_df.to_csv(f"s3://PLACEHOLDER_OUTPUT_PATH/{period}_{state}_training_prediction_validation_output.csv", index=False)
    logging.info("Exported test predictions")

def predict_candidates(period, state, method, model, class_set, pool="all"):
    print("Starting Predicting Candidates...")
    df_cand = pd.read_parquet(f"s3://PLACEHOLDER_PATH/{period}_all_res_training_{state}.parquet").drop(columns = ['STATE_ABBREVIATION', 'CADASTRE_POLYGON_PID', 'CADASTRE_PID'])
    if pool == "all":
        df_cand = df_cand[(df_cand['DS_SOURCE'].str.contains('IMPUTE', case=False, na=False)) | pd.isna(df_cand['DWELLING_STRUCT'])]
    elif pool == "no_ds":
        df_cand = df_cand[pd.isna(df_cand['DS_SOURCE']) | (df_cand['DS_SOURCE'] == "")]
    elif pool == "dsam":
        df_cand = df_cand[df_cand['DS_SOURCE'].str.contains('ADMIN', case=False, na=False)]
    df_og_ds = df_cand[['DWELLING_STRUCT']]
    df_cand = df_cand.drop(columns=['DS_SOURCE', 'DWELLING_STRUCT'])
    df_cand['MESHBLOCK'] = df_cand['MESHBLOCK'].astype("category")
    df_cand['BLD_QUALITY_CLASS'] = df_cand['BLD_QUALITY_CLASS'].astype("category")
    dcand = xgb.DMatrix(df_cand.drop(columns='ARID'), enable_categorical=True)
    prediction_probability = model.predict(dcand)
    probs_df = pd.DataFrame(prediction_probability, columns=class_set),
    probs_df['predicted_class'] = probs_df.idxmax(axis=1)
    probs_df['max_prob'] = probs_df[class_set].max(axis=1)
    probs_df['ARID'] = df_cand.iloc[:, 0].values
    probs_df['original_class'] = df_og_ds.iloc[:,0].values
    probs_df.to_csv(f"s3://PLACEHOLDER_PATH/{period}_{state}_{pool}_predictions.csv", index=False)
    logging.info(f"Exported {period} {state} model for '{pool}' pool.")
    s3_client = boto3.client('s3')
    s3_client.upload_file(Filename=f"model_training_log_{state}.txt", Bucket="PLACEHOLDER", Key=f"PLACEHOLDER/model_training_log_{state}.txt")
    with contextlib.suppress(FileNotFoundError):
        os.remove(f"model_training_log_{state}.txt")
    print(f"Completed predicting candidates for {state}.")
