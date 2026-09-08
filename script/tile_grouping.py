# S2_SuperResolution/utils/tile_grouping.py
#
# cell -> MGRS tile(s) assignment, used by s2dr_availability_schedule to
# batch CDSE queries per tile instead of per cell. The corridor's 435 cells
# (6km grid) actually fall in only 12 distinct Sentinel-2 tiles -- confirmed
# empirically 2026-08-14 by unioning real scene footprints from a live CDSE
# query over the full corridor bbox and intersecting each CELL (not just the
# bbox) against them. (23 tiles intersect the corridor's bounding rectangle,
# but the corridor is a thin winding line, so 11 of those never actually
# cover a real cell -- cell-level intersection is what matters here.) ~22%
# of cells (94/435) sit on a tile boundary and match 2 tiles; those keep
# BOTH, so the schedule can still pick whichever tile's scene is better for
# that cell, rather than silently losing coverage.
#
# Sentinel-2's tiling grid is fixed geographically, so this mapping doesn't
# change day to day -- only when the corridor grid itself is re-materialized
# with different cells. Cached to disk next to corridor_grid.parquet, keyed
# off that file's mtime, so a normal daily schedule tick costs zero extra
# CDSE calls for this step.

import json
from datetime import date, timedelta
from pathlib import Path

from stac_check import _PRODUCTS_URL


def _cache_path(grid_path: str) -> Path:
    return Path(grid_path).with_name("cell_tile_map.json")


def load_or_build_cell_tile_map(
    grid,
    grid_path: str,
    token: str,
    lookback_days: int = 45,
) -> dict[str, list[str]]:
    """Returns {aoi_name: [tile_id, ...]} for every cell in `grid` (a
    DataFrame with bbox_west/south/east/north + aoi_name columns, i.e.
    corridor_grid.parquet). Rebuilds (one CDSE query) only if the cache is
    missing or older than grid_path; otherwise just reads the cache.

    `lookback_days` is wide (45d, vs the schedule's own 10d LOOKBACK_DAYS)
    because this call only needs SOME scene footprint per tile to determine
    geometry -- it doesn't care about cloud cover or recency the way the
    actual per-tile scene selection later does.
    """
    cache = _cache_path(grid_path)
    grid_mtime = Path(grid_path).stat().st_mtime
    if cache.exists() and cache.stat().st_mtime >= grid_mtime:
        return json.loads(cache.read_text())

    import geopandas as gpd
    import requests
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    cells = gpd.GeoDataFrame(
        grid[["aoi_name"]],
        geometry=[
            box(w, s, e, n)
            for w, s, e, n in zip(
                grid.bbox_west, grid.bbox_south, grid.bbox_east, grid.bbox_north
            )
        ],
        crs="EPSG:4326",
    )
    w, s, e, n = cells.total_bounds
    area = f"SRID=4326;POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
    start = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%dT00:00:00.000Z")
    end = date.today().strftime("%Y-%m-%dT23:59:59.999Z")
    filt = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/Value eq 'S2MSI2A') and "
        f"OData.CSC.Intersects(area=geography'{area}') and "
        f"ContentDate/Start gt {start} and ContentDate/Start lt {end}"
    )
    url = f"{_PRODUCTS_URL}?$filter={filt}&$expand=Attributes&$top=1000"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    resp.raise_for_status()
    products = resp.json().get("value", [])

    by_tile_geom: dict[str, list] = {}
    for p in products:
        foot = p.get("GeoFootprint")
        if not foot:
            continue
        tile_id = None
        for attr in p.get("Attributes", []):
            if attr.get("Name") == "tileId":
                tile_id = attr.get("Value")
        if tile_id:
            by_tile_geom.setdefault(tile_id, []).append(shape(foot))

    if not by_tile_geom:
        raise RuntimeError(
            f"tile_grouping: no Sentinel-2 footprints found over the corridor in the "
            f"last {lookback_days} days -- cannot build cell->tile map (widen lookback_days?)"
        )

    tile_union = {t: unary_union(geoms) for t, geoms in by_tile_geom.items()}
    cell_tile_map: dict[str, list[str]] = {}
    for _, row in cells.iterrows():
        hits = [t for t, geom in tile_union.items() if geom.intersects(row.geometry)]
        cell_tile_map[row.aoi_name] = hits

    unmatched = [name for name, tiles in cell_tile_map.items() if not tiles]
    if unmatched:
        raise RuntimeError(
            f"tile_grouping: {len(unmatched)} cell(s) matched no tile in the last "
            f"{lookback_days} days (e.g. {unmatched[:5]}) -- widen lookback_days or "
            "check corridor coverage"
        )

    cache.write_text(json.dumps(cell_tile_map, indent=2))
    return cell_tile_map
