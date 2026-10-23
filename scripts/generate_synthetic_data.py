"""Generate synthetic address / cadastre / building inputs for running training_functions locally.

Mirrors the schemas read by functions/training_functions.py and writes them to data/input/ using the
same folder layout as the S3 placeholders:

    data/input/{period}_dsim/{period}_{training|candidate|all_res}_{state}.parquet
    data/input/{period}_cad/{period}_cad_{state}.parquet
    data/input/{period}_bld/{period}_bld_{state}.parquet

Each grid cell holds one dwelling arrangement whose geometry is typical of its dwelling structure
(separate house, semi, townhouse row, flat blocks of increasing size), so the derived features carry
real signal for the classifier.
"""
import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"

METRIC_CRS = "EPSG:9473"
SOURCE_CRS = "EPSG:7844"
CELL_SIZE = 60.0
MESHBLOCK_CELLS = 5

DWELLING_TYPES = ['x11', 'x21', 'x22', 'x31', 'x32', 'x33', 'x34', 'OTHER']
DWELLING_WEIGHTS = [0.40, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05]
OTHER_CODES = ['x35', 'x91', 'x92', 'x93', 'x94']
# Expected addresses per cell given DWELLING_WEIGHTS and the per-type address counts in build_cell
MEAN_ADDRESSES_PER_CELL = 7.1


class Builder:
    """Collects features column-wise so multi-million record runs stay light on memory."""

    def __init__(self, rng):
        self.rng = rng
        self.parcels = {'geometry': [], 'MB_CODE_2021': []}
        self.buildings = {'geometry': [], 'MB_CODE_2021': [], 'QUALITY_CLASS': [], 'ADDRESS_COUNT': []}
        self.addresses = {'x': [], 'y': [], 'MESHBLOCK': [], 'TRUE_CLASS': []}

    @property
    def n_addresses(self):
        return len(self.addresses['x'])

    def parcel(self, geom, meshblock):
        self.parcels['geometry'].append(geom)
        self.parcels['MB_CODE_2021'].append(meshblock)

    def building(self, geom, meshblock, quality, n_addresses):
        self.buildings['geometry'].append(geom)
        self.buildings['MB_CODE_2021'].append(meshblock)
        self.buildings['QUALITY_CLASS'].append(quality)
        self.buildings['ADDRESS_COUNT'].append(n_addresses)

    def address(self, x, y, meshblock, dwelling, n=1):
        a = self.addresses
        if np.ndim(x) == 0:
            x, y = [x] * n, [y] * n
        a['x'].extend(x)
        a['y'].extend(y)
        a['MESHBLOCK'].extend([meshblock] * n)
        a['TRUE_CLASS'].extend([dwelling] * n)

    def jitter(self, scale=1.0):
        return self.rng.uniform(-scale, scale, 2)

    def addresses_in(self, geom, n, meshblock, dwelling, shared_geocode=False):
        minx, miny, maxx, maxy = geom.bounds
        if shared_geocode:
            dx, dy = self.jitter()
            self.address((minx + maxx) / 2 + dx, (miny + maxy) / 2 + dy, meshblock, dwelling, n)
        else:
            xs = self.rng.uniform(minx + 0.5, maxx - 0.5, n)
            ys = self.rng.uniform(miny + 0.5, maxy - 0.5, n)
            self.address(xs.tolist(), ys.tolist(), meshblock, dwelling, n)


def build_cell(b, x0, y0, meshblock, dwelling):
    rng = b.rng
    if dwelling == 'x11':
        b.parcel(box(x0, y0, x0 + 18, y0 + 35), meshblock)
        w, h = rng.uniform(10, 14), rng.uniform(13, 18)
        house = box(x0 + 3, y0 + 8, x0 + 3 + w, y0 + 8 + h)
        b.building(house, meshblock, rng.choice(['A', 'B', 'C'], p=[0.5, 0.35, 0.15]), 1)
        if rng.random() < 0.3:
            b.building(box(x0 + 13, y0 + 29, x0 + 16, y0 + 32), meshblock, 'C', 0)
        if rng.random() < 0.1:
            # Geocoded on the street frontage, outside the parcel
            b.address(x0 + 9, y0 - 3, meshblock, dwelling)
        else:
            dx, dy = b.jitter()
            b.address(x0 + 3 + w / 2 + dx, y0 + 8 + h / 2 + dy, meshblock, dwelling)

    elif dwelling == 'x21':
        h = rng.uniform(12, 16)
        left = box(x0 + 2, y0 + 8, x0 + 10, y0 + 8 + h)
        right = box(x0 + 10, y0 + 8, x0 + 18, y0 + 8 + h)
        if rng.random() < 0.5:
            b.parcel(box(x0, y0, x0 + 20, y0 + 35), meshblock)
        else:
            # Strata split down the party wall, with the left half slightly over the line
            left = box(x0 + 2, y0 + 8, x0 + 10.5, y0 + 8 + h)
            right = box(x0 + 10.5, y0 + 8, x0 + 18, y0 + 8 + h)
            b.parcel(box(x0, y0, x0 + 10, y0 + 35), meshblock)
            b.parcel(box(x0 + 10, y0, x0 + 20, y0 + 35), meshblock)
        for unit in (left, right):
            b.building(unit, meshblock, rng.choice(['A', 'B'], p=[0.6, 0.4]), 1)
            b.addresses_in(unit, 1, meshblock, dwelling)

    elif dwelling == 'x22':
        b.parcel(box(x0, y0, x0 + 42, y0 + 35), meshblock)
        n_units = rng.integers(3, 6)
        width = 40 / n_units
        for i in range(n_units):
            unit = box(x0 + 1 + i * width, y0 + 10, x0 + 1 + (i + 1) * width, y0 + 22)
            b.building(unit, meshblock, 'B', 1)
            b.addresses_in(unit, 1, meshblock, dwelling)

    elif dwelling == 'OTHER':
        b.parcel(box(x0, y0, x0 + 40, y0 + 40), meshblock)
        w, h = rng.uniform(6, 30, 2)
        bld = box(x0 + 5, y0 + 5, x0 + 5 + w, y0 + 5 + h)
        n = int(rng.integers(1, 4))
        b.building(bld, meshblock, rng.choice(['A', 'B', 'C']), n)
        b.addresses_in(bld, n, meshblock, rng.choice(OTHER_CODES), shared_geocode=rng.random() < 0.5)

    else:
        # Flats: building size and address count grow with the class
        spec = {
            'x31': (30, 35, (14, 20), (2, 5), 0.8),
            'x32': (40, 40, (20, 28), (5, 12), 0.3),
            'x33': (48, 48, (28, 36), (10, 30), 0.3),
            'x34': (50, 50, (36, 44), (25, 70), 0.6),
        }[dwelling]
        pw, ph, (smin, smax), (nmin, nmax), p_shared = spec
        b.parcel(box(x0, y0, x0 + pw, y0 + ph), meshblock)
        s = rng.uniform(smin, smax)
        bld = box(x0 + 2, y0 + 2, x0 + 2 + min(s, pw - 4), y0 + 2 + min(s, ph - 4))
        n = int(rng.integers(nmin, nmax + 1))
        b.building(bld, meshblock, rng.choice(['A', 'B'], p=[0.7, 0.3]), n)
        b.addresses_in(bld, n, meshblock, dwelling, shared_geocode=rng.random() < p_shared)

    if rng.random() < 0.2:
        # Small easement / reserve parcel, filtered out by the CAD_AREA > 100 rule
        b.parcel(box(x0 + 50, y0 + 45, x0 + 58, y0 + 55), meshblock)


def to_wkt(geoms):
    projected = gpd.GeoSeries(geoms, crs=METRIC_CRS).to_crs(SOURCE_CRS)
    return shapely.to_wkt(projected.array, rounding_precision=9)


def generate(period, state, grid, seed, records=None):
    rng = np.random.default_rng(seed)
    origin = gpd.GeoSeries([Point(138.60, -34.93)], crs=SOURCE_CRS).to_crs(METRIC_CRS).iloc[0]
    b = Builder(rng)
    target = None
    if records:
        # Stop filling the grid once the target is reached; 1% is added later as duplicate ARIDs
        target = records / 1.01
        grid = math.ceil(math.sqrt(target / MEAN_ADDRESSES_PER_CELL) * 1.05)
    dwellings = rng.choice(DWELLING_TYPES, size=(grid, grid), p=DWELLING_WEIGHTS)
    for i in range(grid):
        for j in range(grid):
            meshblock = f"4{(i // MESHBLOCK_CELLS):05d}{(j // MESHBLOCK_CELLS):05d}"
            build_cell(b, origin.x + i * CELL_SIZE, origin.y + j * CELL_SIZE, meshblock, dwellings[i, j])
        if target and b.n_addresses >= target:
            break

    # Cadastre
    cad = pd.DataFrame(b.parcels)
    cad_geoms = cad.pop('geometry')
    cad.insert(0, 'CADASTRE_POLYGON_PID', [f"cp{n:08d}" for n in range(len(cad))])
    cad.insert(1, 'CADASTRE_PID', [f"cd{n:08d}" for n in range(len(cad))])
    cad['CAD_AREA'] = shapely.area(np.asarray(cad_geoms))
    cad['CAD_POLY_PID_COUNT'] = 1
    cad['CAD_POLY'] = to_wkt(cad_geoms)

    # Buildings
    bld = pd.DataFrame(b.buildings)
    bld_geoms = bld.pop('geometry')
    bld.insert(0, 'BUILDING_PID', [f"bl{n:08d}" for n in range(len(bld))])
    bld['BUILDING_NUM_VERTICES'] = 5
    bld['BUILDING_AREA'] = shapely.area(np.asarray(bld_geoms))
    bld['BLD_POLY'] = to_wkt(bld_geoms)

    # Addresses
    ar = pd.DataFrame(b.addresses)
    del b
    ar_points = gpd.GeoSeries(gpd.points_from_xy(ar.pop('x'), ar.pop('y')), crs=METRIC_CRS).to_crs(SOURCE_CRS)
    true_class = ar.pop('TRUE_CLASS').to_numpy()
    n = len(ar)
    ar.insert(0, 'ARID', [f"ar{k:09d}" for k in range(n)])
    ar['LATITUDE'] = ar_points.y.to_numpy()
    ar['LONGITUDE'] = ar_points.x.to_numpy()
    ar['STATE_ABBREVIATION'] = state
    # 8% label noise so the model has something to get wrong
    noisy = rng.random(n) < 0.08
    labels = np.where(noisy, rng.choice(DWELLING_TYPES[:-1] + OTHER_CODES[:1], size=n), true_class)
    ar['DS_SOURCE'] = rng.choice(['DCAN_2021', 'IMPUTE_MODEL', 'ADMIN_RECORD', None], size=n, p=[0.6, 0.2, 0.1, 0.1])
    ar['DWELLING_STRUCT'] = np.where(ar['DS_SOURCE'].isna(), None, labels)
    # A few repeated ARIDs, which the loader collapses with groupby('ARID').first()
    ar = pd.concat([ar, ar.sample(frac=0.01, random_state=seed)], ignore_index=True)

    dsim_dir = INPUT_DIR / f"{period}_dsim"
    cad_dir = INPUT_DIR / f"{period}_cad"
    bld_dir = INPUT_DIR / f"{period}_bld"
    for d in (dsim_dir, cad_dir, bld_dir):
        d.mkdir(parents=True, exist_ok=True)

    is_dcan = ar['DS_SOURCE'].str.contains('DCAN', na=False)
    ar.to_parquet(dsim_dir / f"{period}_all_res_{state}.parquet", index=False)
    ar[is_dcan].to_parquet(dsim_dir / f"{period}_training_{state}.parquet", index=False)
    ar[~is_dcan].to_parquet(dsim_dir / f"{period}_candidate_{state}.parquet", index=False)
    cad.to_parquet(cad_dir / f"{period}_cad_{state}.parquet", index=False)
    bld.to_parquet(bld_dir / f"{period}_bld_{state}.parquet", index=False)
    print(f"Wrote {len(ar):,} addresses, {len(cad):,} parcels, {len(bld):,} buildings to {INPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", default="a25")
    parser.add_argument("--state", default="SA")
    parser.add_argument("--grid", type=int, default=60, help="cells per side; grid**2 dwelling arrangements")
    parser.add_argument("--records", type=int, help="target number of address records (overrides --grid)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.period, args.state, args.grid, args.seed, args.records)
