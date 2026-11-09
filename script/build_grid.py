"""Generic AOI -> grid builder for an ad-hoc s2sr-corridor run.

Not used by the production `daily_run.py` (which has its own fixed
grid-7km5/corridor_grid.parquet) -- this is for anyone cloning this repo to
run against their OWN AOI, per the README's "Building your own grid"
section.

Tiles the AOI polygon into uniform patch_km-sized squares, keeps only cells
that intersect the true AOI (not just its bbox), and writes:
  <out_dir>/grid.parquet                       (cell_id, bbox_*, patch_km, aoi_name, aoi_file)
  <out_dir>/aoi_cells/<prefix>-NNNN.geojson     (one per kept cell, WGS84)

cell_km is deliberately smaller than patch_km (default 7.5 vs 8.0) so a cell
sits safely inside one patch with margin -- see the README's note on why an
AOI the same size as (or larger than) patch_km risks the model rounding it
up into extra patches.

Usage:
    python build_grid.py AOI.shp OUT_DIR [--prefix NAME] [--cell-km 7.5] [--patch-km 8.0]

AOI.shp can be any format geopandas reads (Shapefile, GeoPackage, GeoJSON).
Its CRS can be anything projected or geographic -- cells are built in the
AOI's own CRS for exact square cells, then reprojected to WGS84 for output
(the vendor image expects WGS84 --aoi-file input regardless of source CRS).
"""
from __future__ import annotations

import argparse
import json
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def build_grid(aoi_path: str, out_dir: str, prefix: str = "CELL",
                cell_km: float = 7.5, patch_km: float = 8.0) -> pd.DataFrame:
    if cell_km >= patch_km:
        raise ValueError(
            f"cell_km ({cell_km}) must be smaller than patch_km ({patch_km}) "
            "-- a cell the same size as or larger than one patch risks the "
            "model rounding it up into multiple patches (see README)."
        )

    aoi = gpd.read_file(aoi_path)
    working_crs = aoi.crs if aoi.crs and aoi.crs.is_projected else aoi.estimate_utm_crs()
    aoi = aoi.to_crs(working_crs)
    union = aoi.union_all()
    minx, miny, maxx, maxy = aoi.total_bounds

    cells_dir = os.path.join(out_dir, "aoi_cells")
    os.makedirs(cells_dir, exist_ok=True)

    step_m = cell_km * 1000
    records = []
    cell_id = 0
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cell = box(x, y, x + step_m, y + step_m)
            if cell.intersects(union):
                aoi_name = f"{prefix}-{cell_id:04d}"
                geom_wgs84 = gpd.GeoSeries([cell], crs=working_crs).to_crs("EPSG:4326").iloc[0]
                aoi_file = os.path.join(cells_dir, f"{aoi_name}.geojson")
                with open(aoi_file, "w") as f:
                    json.dump({
                        "type": "FeatureCollection",
                        "features": [{
                            "type": "Feature",
                            "properties": {"aoi_name": aoi_name},
                            "geometry": geom_wgs84.__geo_interface__,
                        }],
                    }, f)
                b = geom_wgs84.bounds
                records.append({
                    "cell_id": cell_id, "aoi_name": aoi_name, "aoi_file": aoi_file,
                    "patch_km": patch_km,
                    "bbox_west": b[0], "bbox_south": b[1], "bbox_east": b[2], "bbox_north": b[3],
                })
                cell_id += 1
            x += step_m
        y += step_m

    df = pd.DataFrame(records)
    df.to_parquet(os.path.join(out_dir, "grid.parquet"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aoi_path", help="AOI boundary file (any geopandas-readable format)")
    ap.add_argument("out_dir", help="Directory to write grid.parquet + aoi_cells/ into")
    ap.add_argument("--prefix", default="CELL", help="aoi_name prefix, e.g. MYPROJECT -> MYPROJECT-0000 (default: CELL)")
    ap.add_argument("--cell-km", type=float, default=7.5, help="Cell size in km (default: 7.5)")
    ap.add_argument("--patch-km", type=float, default=8.0, help="Patch size in km, must be > cell-km (default: 8.0)")
    args = ap.parse_args()

    result = build_grid(args.aoi_path, args.out_dir, args.prefix, args.cell_km, args.patch_km)
    print(f"wrote {len(result)} cells -> {args.out_dir}/grid.parquet")
