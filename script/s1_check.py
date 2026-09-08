# S2_SuperResolution/utils/s1_check.py
#
# "Which corridor cells does today's Sentinel-1 pass cover?" -- used to
# decide the PRIORITY cell set for a day's S2DR run (process the segment
# that has fresh S1 coverage first, then the rest of the corridor).
#
# S1 and S2 are independent constellations on independent revisit cycles --
# a day with zero S1 passes near the corridor is normal and does NOT mean
# no S2 imagery that day (see 2026-08-18: 0 S1 scenes, but S2 still covered
# 47/47 cells of the corridor's orbit99 segment). When today has no S1
# pass, this looks back up to LOOKBACK_DAYS for the most recent one and
# uses that orbit's footprint as the priority reference instead.
#
# Queries ASF's public search API directly (metadata only, no auth
# required for search) -- same approach validated against real data
# 2026-08-14/17.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import requests

_ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"

# Buffer added on each side of the S1 swath footprint, in the grid's own
# metric CRS units (meters) -- matches one 6km cell width, so a cell that's
# geometrically adjacent to (not just directly under) the S1 pass still
# counts as priority. Confirmed empirically 2026-08-17/19: this typically
# adds ~10-15% more cells on top of the direct-intersect count.
PRIORITY_BUFFER_M = 6000

LOOKBACK_DAYS_DEFAULT = 7


@dataclass
class S1OrbitHit:
    orbit: int
    direction: str
    n_scenes: int
    direct_cells: int
    buffered_cells: int
    s2_tiles: list[str] = field(default_factory=list)


@dataclass
class S1CheckResult:
    check_date: str
    s1_reference_date: str | None  # None if no S1 pass found even after lookback
    orbits_found: list[S1OrbitHit]
    priority_cell_ids: list[int]  # union across all orbits found that date, deduped
    no_s1_found: bool


def _query_asf(bbox_wkt_polygon: str, date_str: str) -> list[dict]:
    params = {
        "platform": "Sentinel-1",
        "processingLevel": "SLC",
        "beamMode": "IW",
        "start": f"{date_str}T00:00:00Z",
        "end": f"{date_str}T23:59:59Z",
        "intersectsWith": bbox_wkt_polygon,
        "output": "json",
    }
    resp = requests.get(_ASF_SEARCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else []


def _corridor_bbox_wkt(cells) -> str:
    """cells: a GeoDataFrame (WGS84) covering the corridor grid."""
    w, s, e, n = cells.total_bounds
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def find_s1_reference_date(
    cells,
    check_date: date,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> tuple[str | None, list[dict]]:
    """Returns (reference_date_str, raw_scene_list) for the most recent day
    (check_date going backwards) that had an S1 pass near the corridor.
    (None, []) if nothing in the lookback window."""
    bbox_wkt = _corridor_bbox_wkt(cells)
    for delta in range(lookback_days + 1):
        d = check_date - timedelta(days=delta)
        d_str = d.isoformat()
        scenes = _query_asf(bbox_wkt, d_str)
        if scenes:
            return d_str, scenes
    return None, []


def check(cells, tile_map: dict, check_date: date, lookback_days: int = LOOKBACK_DAYS_DEFAULT) -> S1CheckResult:
    """Main entry point.

    cells: GeoDataFrame (WGS84) with cell_id + geometry, the 6km corridor grid
           (or a subset), reprojectable to a metric CRS for buffering.
    tile_map: {aoi_name: [tile_id, ...]} from tile_grouping.load_or_build_cell_tile_map,
              used only to report which S2 tiles a priority orbit corresponds to.
    """
    from shapely import wkt as shapely_wkt
    from shapely.ops import unary_union

    ref_date_str, scenes = find_s1_reference_date(cells, check_date, lookback_days)
    if ref_date_str is None:
        return S1CheckResult(
            check_date=check_date.isoformat(),
            s1_reference_date=None,
            orbits_found=[],
            priority_cell_ids=[],
            no_s1_found=True,
        )

    cells_m = cells.to_crs(32647)
    by_orbit: dict[tuple[int, str], list[dict]] = {}
    for sc in scenes:
        key = (sc.get("relativeOrbit") or sc.get("pathNumber"), sc.get("flightDirection"))
        by_orbit.setdefault(key, []).append(sc)

    orbit_hits: list[S1OrbitHit] = []
    all_priority_ids: set[int] = set()
    for (orbit, direction), orbit_scenes in by_orbit.items():
        geoms = [shapely_wkt.loads(sc["stringFootprint"]) for sc in orbit_scenes if sc.get("stringFootprint")]
        if not geoms:
            continue
        footprint_m = _to_metric(unary_union(geoms))
        direct = cells_m[cells_m.intersects(footprint_m)]
        buffered_geom = footprint_m.buffer(PRIORITY_BUFFER_M)
        buffered = cells_m[cells_m.intersects(buffered_geom)]

        cell_ids = set(buffered["cell_id"].tolist())
        all_priority_ids |= cell_ids

        aoi_names = buffered["aoi_name"].tolist() if "aoi_name" in buffered.columns else []
        tiles = sorted({t for name in aoi_names for t in tile_map.get(name, [])})

        orbit_hits.append(S1OrbitHit(
            orbit=int(orbit) if orbit is not None else -1,
            direction=direction or "UNKNOWN",
            n_scenes=len(orbit_scenes),
            direct_cells=len(direct),
            buffered_cells=len(buffered),
            s2_tiles=tiles,
        ))

    return S1CheckResult(
        check_date=check_date.isoformat(),
        s1_reference_date=ref_date_str,
        orbits_found=orbit_hits,
        priority_cell_ids=sorted(all_priority_ids),
        no_s1_found=False,
    )


def _to_metric(geom_wgs84):
    import geopandas as gpd
    return gpd.GeoSeries([geom_wgs84], crs=4326).to_crs(32647).iloc[0]


def to_metadata_dict(result: S1CheckResult, s2_covered_cell_count: int | None = None) -> dict:
    """The log/metadata shape agreed 2026-08-19 -- see s2dr_daily_run_asset.py."""
    d = {
        "check_date": result.check_date,
        "s1_reference_date": result.s1_reference_date,
        "no_s1_found": result.no_s1_found,
        "s1_orbits_found": [
            {
                "orbit": h.orbit,
                "direction": h.direction,
                "n_scenes": h.n_scenes,
                "direct_cells": h.direct_cells,
                "buffered_cells": h.buffered_cells,
                "s2_tiles": h.s2_tiles,
            }
            for h in result.orbits_found
        ],
        "priority_cell_count": len(result.priority_cell_ids),
    }
    if s2_covered_cell_count is not None:
        d["s2_covered_cell_count"] = s2_covered_cell_count
        d["remaining_cell_count"] = max(0, s2_covered_cell_count - len(result.priority_cell_ids))
    return d
