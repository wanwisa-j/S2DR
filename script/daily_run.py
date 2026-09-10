# S2_SuperResolution/utils/daily_run.py
#
# Priority/full daily corridor run -- the logic agreed 2026-08-19:
#
#   mode="priority": s1_check finds today's (or the most recent) S1 pass,
#     maps it to corridor cells (+1 cell buffer), and processes ONLY that
#     set. Saves output/{date}/priority/corridor_{date}_priority.vrt.
#
#   mode="full": processes every corridor cell with real S2 coverage that
#     date. Cells already done (output_already_done -- same dedup core_runner
#     already uses) are skipped, so running `full` after `priority` (or
#     after a partial previous `full`) naturally only does the remaining
#     work, never redoes anything. Saves
#     output/{date}/full/corridor_{date}_full.vrt once every cell in the
#     set has been attempted.
#
# Both modes share the SAME dynamic auto-concurrency worker pool
# (capacity_probe.probe_host, re-checked every 30s) -- there is no separate
# "priority worker count" vs "full worker count", only queue ORDER differs.
#
# This module is meant to be called both by the Dagster asset (UI: date +
# mode) and directly from a script for standalone testing before that
# wiring exists (per 2026-08-19 discussion: "ครบ loop ก่อน ให้เรารันใน
# dagster ได้" -- prove the loop works stand-alone first).

from __future__ import annotations

import os
import queue
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import date as date_cls

from capacity_probe import probe_host
from cloud_mask import cloud_fraction_over, find_omni_scl, load_scl, mask_cloud_nodata
from core_runner import (
    remote_date_dir, run_s2_super_resolution, validate_output, output_already_done,
)
import s1_check
from stac_check import _PRODUCTS_URL, _get_cdse_token
from tile_grouping import load_or_build_cell_tile_map

HOST_ID = "87"
PATCH_KM = 8.0
REPROBE_INTERVAL_S = 30
GRID_PATH = "/fs2/wanwisa/s2sr-corridor/grid-7km5/corridor_grid.parquet"

# Cloud pre-filter: skip a cell BEFORE dispatch when this fraction or more of
# its TRUE corridor shape (not the 6km bbox) is cloud ({1,2} in OMNI_SCL) --
# such a cell delivers a raster mask_cloud_nodata nulls out almost entirely, so
# the ~250s GPU run is wasted. Deliberately high: a cell with any usable clear
# corridor still runs. MIN_PX guards the estimate on the ~7px-wide strip.
CLOUD_SKIP_FRACTION = 0.95
CLOUD_SKIP_MIN_PX = 200

# Per-cell retry: the container skips a patch on transient source/read faults
# (corrupt COG chunk, RasterioIOError, HTTP hiccup) and then fails the whole
# cell with "no patches produced". Re-running the docker-run is a CLEAN retry --
# fresh /content/datapath, fresh download -- so a corrupt-chunk failure like
# `IMCD_LZW_CORRUPT` (seen on cell 00083, 2026-09-09) usually succeeds the
# second time. A non-transient failure (bad AOI, real OOM) is not retried.
CELL_RETRY_ATTEMPTS = 2
_TRANSIENT_MARKERS = (
    "IMCD_LZW", "imcd_lzw", "CORRUPT", "RasterioIOError", "Error opening",
    "HTTP response code", "timed out", "Connection", "CURL", "CPLE_",
)
# Circuit breaker (fastsar "fail closed, not guess-and-continue"): if this many
# cells in a row fail with a transient/IO fault, the shared filesystem or S3 is
# probably in trouble -- stop dispatching rather than grind through the rest of
# the corridor hammering a sick NAS (the 2026-09-02 /fs2 incident).
IO_CIRCUIT_BREAK_CONSECUTIVE = 8


def _sleep_with_backoff(attempt: int, base: float = 3.0, cap: float = 60.0) -> float:
    """Exponential backoff + 0-25% jitter (so concurrent workers don't retry in
    lockstep against the same S3/NAS). Ported from sentinel-1-fastsar
    downloader/retry_util.py. Returns the delay slept."""
    delay = min(cap, base * (2 ** (attempt - 1)))
    delay += random.uniform(0, delay * 0.25)
    time.sleep(delay)
    return delay


def _is_transient(msg: str | None) -> bool:
    return bool(msg) and any(m in msg for m in _TRANSIENT_MARKERS)


def _cell_geom(aoi_file: str):
    """Shapely polygon (WGS84) of a corridor cell's aoi_file GeoJSON."""
    import json

    from shapely.geometry import shape
    from shapely.ops import unary_union

    with open(aoi_file) as fh:
        gj = json.load(fh)
    feats = gj.get("features", [gj])
    return unary_union([shape(f["geometry"]) for f in feats])


def select_covered_cells(date_str: str, token: str | None = None):
    """Every 6km corridor cell with real, confirmed S2 coverage on date_str,
    each with its best-covering tile_id + cloud_cover attached (tile_id is
    required for cloud_mask). Same method as the validated
    select_corridor_cells.py.

    Returns (records, cells_gdf): records is a list[dict] of every covered
    cell (cell_id, aoi_name, aoi_file, tile_id, cloud_cover); cells_gdf is
    the FULL 435-cell grid (WGS84, unfiltered) -- s1_check needs the full
    grid, not just the covered subset, to compute priority cell membership
    correctly."""
    import geopandas as gpd
    import pandas as pd
    import requests
    from shapely.geometry import box, shape

    if token is None:
        token = _get_cdse_token()

    grid = pd.read_parquet(GRID_PATH)
    cells = gpd.GeoDataFrame(
        grid[["cell_id", "aoi_name", "aoi_file"]],
        geometry=[
            box(w, s, e, n)
            for w, s, e, n in zip(grid.bbox_west, grid.bbox_south, grid.bbox_east, grid.bbox_north)
        ],
        crs="EPSG:4326",
    )
    w, s, e, n = cells.total_bounds
    area = f"SRID=4326;POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
    filt = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/Value eq 'S2MSI2A') and "
        f"OData.CSC.Intersects(area=geography'{area}') and "
        f"ContentDate/Start gt {date_str}T00:00:00.000Z and ContentDate/Start lt {date_str}T23:59:59.999Z"
    )
    url = f"{_PRODUCTS_URL}?$filter={filt}&$expand=Attributes&$top=1000"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    resp.raise_for_status()
    products = resp.json().get("value", [])

    scene_info = []
    for p in products:
        foot = p.get("GeoFootprint")
        if not foot:
            continue
        tile_id, cloud = None, None
        for attr in p.get("Attributes", []):
            if attr.get("Name") == "tileId":
                tile_id = attr.get("Value")
            if attr.get("Name") == "cloudCover":
                cloud = attr.get("Value")
        scene_info.append({"geom": shape(foot), "tile_id": tile_id, "cloud_cover": cloud})

    def tile_for_cell(cell_geom):
        hits = [si for si in scene_info if si["geom"].intersects(cell_geom)]
        hits = [h for h in hits if h["cloud_cover"] is not None]
        if not hits:
            return None, None
        best = min(hits, key=lambda h: h["cloud_cover"])
        return best["tile_id"], best["cloud_cover"]

    records = []
    for _, row in cells.iterrows():
        tile_id, cloud = tile_for_cell(row.geometry)
        if tile_id is None:
            continue
        records.append({
            "cell_id": int(row.cell_id),
            "aoi_name": row.aoi_name,
            "aoi_file": row.aoi_file,
            "tile_id": tile_id,
            "cloud_cover": cloud,
        })
    return records, cells  # cells returned too, for s1_check's buffering


@dataclass
class DailyRunResult:
    date_str: str
    mode: str
    s1_metadata: dict
    n_selected: int
    n_ok: int
    n_skipped_already_done: int
    total_wall_time_s: float
    vrt_path: str | None
    n_skipped_cloud: int = 0
    patches: list[dict] = field(default_factory=list)


def _build_vrt(date_str: str, subdir: str, aoi_names: list[str]) -> str | None:
    out_dir = remote_date_dir(date_str)
    vrt_dir = os.path.join(out_dir, subdir)
    os.makedirs(vrt_dir, exist_ok=True)
    date_compact = date_str.replace("-", "")

    files = []
    for name in aoi_names:
        f = os.path.join(out_dir, name, f"S2SR10x_{name}_{date_compact}_TCI_1m.tif")
        if os.path.exists(f):
            files.append(os.path.relpath(f, vrt_dir))
    if not files:
        return None

    vrt_name = f"corridor_{date_compact}_{subdir}.vrt"
    subprocess.run(["gdalbuildvrt", vrt_name, *files], cwd=vrt_dir, check=True, capture_output=True)
    return os.path.join(vrt_dir, vrt_name)


def run_daily(
    date_str: str,
    mode: str,
    host_id: str = HOST_ID,
    patch_km: float = PATCH_KM,
    progress_cb=None,
) -> DailyRunResult:
    """progress_cb, if given, is called twice per phase before work starts,
    then once per cell (including skips) during it -- always
    progress_cb(event: dict), branch on event["phase"]:
      "s2_coverage"  -- {n_covered}: cells with real S2 coverage today,
                        before S1 is even checked.
      "s1_check"     -- {s1_meta}: the dict from s1_check.to_metadata_dict
                        (orbit/direction/n_scenes/cells/tiles found).
      "work_selected"-- {mode, n_selected}: final cell count for this run
                        after mode (priority/full) narrows s2_coverage.
      "cell_dispatch_start" -- {aoi_name, worker_id}: fires right before
                        the blocking SSH docker-run call for one cell --
                        distinguishes "worker hasn't reached SSH yet" from
                        "SSH call itself is just slow" (added 2026-08-19
                        after a run looked hung with 0 completions for 12+
                        minutes; turned out to be a bad SSH_KEY_87 path in
                        .env, silently failing capacity_probe every 30s
                        reprobe with no per-cell log to show it).
      (no "phase" key) -- per-cell: aoi_name, skipped, returncode,
                        wall_time_s, n_done, n_total.
    These fire early -- select_covered_cells + s1_check.check
    are both live API calls (CDSE, ASF) that can take a while with nothing
    else to show for it otherwise. Callers can print() (standalone testing)
    or context.log.info() (Dagster) with it; this module stays framework-
    agnostic."""
    if mode not in ("priority", "full"):
        raise ValueError(f"mode must be 'priority' or 'full', got {mode!r}")

    import pandas as pd

    check_date = date_cls.fromisoformat(date_str)
    covered_cells, cells_gdf = select_covered_cells(date_str)
    if progress_cb:
        progress_cb({"phase": "s2_coverage", "n_covered": len(covered_cells)})

    token = _get_cdse_token()
    full_grid = pd.read_parquet(GRID_PATH)
    tile_map = load_or_build_cell_tile_map(full_grid, GRID_PATH, token)
    s1_result = s1_check.check(cells_gdf, tile_map, check_date)
    s1_meta = s1_check.to_metadata_dict(s1_result, s2_covered_cell_count=len(covered_cells))
    if progress_cb:
        progress_cb({"phase": "s1_check", "s1_meta": s1_meta})

    if mode == "priority":
        priority_ids = set(s1_result.priority_cell_ids)
        work_cells = [c for c in covered_cells if c["cell_id"] in priority_ids]
        vrt_subdir = "priority"
    else:
        work_cells = covered_cells
        vrt_subdir = "full"

    if progress_cb:
        progress_cb({"phase": "work_selected", "mode": mode, "n_selected": len(work_cells)})

    work_q: queue.Queue = queue.Queue()
    for c in work_cells:
        work_q.put(c)

    results: list[dict] = []
    results_lock = threading.Lock()
    n_skipped = 0
    n_skipped_cloud = 0
    io_fail_streak = 0      # consecutive transient/IO cell failures (circuit breaker)
    io_circuit_open = False

    def worker(worker_id: int):
        nonlocal n_skipped, n_skipped_cloud, io_fail_streak, io_circuit_open
        while True:
            if io_circuit_open:
                return
            try:
                cell = work_q.get_nowait()
            except queue.Empty:
                return
            aoi_name = cell["aoi_name"]
            tile_id = cell["tile_id"]

            if mode == "full" and output_already_done(aoi_name, date_str):
                with results_lock:
                    n_skipped += 1
                    results.append({"aoi_name": aoi_name, "skipped": True})
                    n_done = len(results)
                if progress_cb:
                    progress_cb({"aoi_name": aoi_name, "skipped": True, "returncode": None,
                                 "wall_time_s": None, "n_done": n_done, "n_total": len(work_cells)})
                continue

            # Cloud pre-filter -- read this cell's SCL ONCE, reused for the
            # post-inference mask below. Never drops a cell on its own error.
            scl = None
            cloud_frac = None
            omni = find_omni_scl(tile_id, date_str)
            if omni:
                try:
                    scl = load_scl(omni)
                    cloud_frac, n_px = cloud_fraction_over(scl, _cell_geom(cell["aoi_file"]))
                    if n_px >= CLOUD_SKIP_MIN_PX and cloud_frac >= CLOUD_SKIP_FRACTION:
                        with results_lock:
                            n_skipped_cloud += 1
                            results.append({"aoi_name": aoi_name, "skipped": True,
                                            "skip_reason": "cloud",
                                            "cloud_fraction": round(cloud_frac, 3)})
                            n_done = len(results)
                        if progress_cb:
                            progress_cb({"aoi_name": aoi_name, "skipped": True,
                                         "returncode": None, "wall_time_s": None,
                                         "n_done": n_done, "n_total": len(work_cells),
                                         "skip_reason": "cloud",
                                         "cloud_fraction": round(cloud_frac, 3)})
                        continue
                except Exception:  # noqa: BLE001 -- pre-filter must not lose a cell
                    scl = None

            t0 = time.time()
            error, returncode, n_attempts = None, None, 0
            if progress_cb:
                progress_cb({"phase": "cell_dispatch_start", "aoi_name": aoi_name,
                             "worker_id": worker_id})
            for attempt in range(1, CELL_RETRY_ATTEMPTS + 1):
                n_attempts = attempt
                try:
                    result = run_s2_super_resolution(
                        host_id=host_id, aoi_file=cell["aoi_file"], date_str=date_str,
                        aoi_name=aoi_name, patch_km=patch_km, gpu_available=True, threads=4,
                    )
                    returncode = result.returncode
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001 -- one bad cell must not kill the run
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < CELL_RETRY_ATTEMPTS and _is_transient(error):
                        slept = _sleep_with_backoff(attempt)
                        if progress_cb:
                            progress_cb({"phase": "cell_retry", "aoi_name": aoi_name,
                                         "worker_id": worker_id, "attempt": attempt,
                                         "backoff_s": round(slept, 1), "error": error[:120]})
                        continue
                    break
            t1 = time.time()

            output_path, validation_error = None, None
            try:
                p = validate_output(aoi_name, date_str)
                output_path = str(p)
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"

            # Circuit breaker (fastsar "fail closed"): a clean success resets the
            # streak; a transient failure that survived the retry extends it, and
            # once IO_CIRCUIT_BREAK_CONSECUTIVE cells in a row have failed that
            # way, stop dispatching the rest -- the shared FS / S3 is likely
            # degrading and grinding on only makes it worse.
            with results_lock:
                if output_path and returncode == 0:
                    io_fail_streak = 0
                elif _is_transient(error):
                    io_fail_streak += 1
                    if io_fail_streak >= IO_CIRCUIT_BREAK_CONSECUTIVE and not io_circuit_open:
                        io_circuit_open = True
                        while not work_q.empty():
                            try:
                                work_q.get_nowait()
                            except queue.Empty:
                                break
                        if progress_cb:
                            progress_cb({"phase": "io_circuit_open",
                                         "consecutive_failures": io_fail_streak})

            cloud_mask_result, cloud_mask_error = None, None
            if output_path:
                try:
                    if scl is None and omni:  # pre-filter couldn't build it; retry
                        scl = load_scl(omni)
                    if scl is not None:
                        cloud_mask_result = mask_cloud_nodata(output_path, scl)
                    else:
                        cloud_mask_error = "no OMNI_SCL found"
                except Exception as exc:
                    cloud_mask_error = f"{type(exc).__name__}: {exc}"

            with results_lock:
                results.append({
                    "worker_id": worker_id, "aoi_name": aoi_name, "skipped": False,
                    "wall_time_s": round(t1 - t0, 2), "returncode": returncode,
                    "n_attempts": n_attempts, "run_error": error, "output_path": output_path,
                    "validation_error": validation_error,
                    "cloud_fraction": round(cloud_frac, 3) if cloud_frac is not None else None,
                    "cloud_mask_result": cloud_mask_result, "cloud_mask_error": cloud_mask_error,
                })
                n_done = len(results)
            if progress_cb:
                progress_cb({"aoi_name": aoi_name, "skipped": False, "returncode": returncode,
                             "wall_time_s": round(t1 - t0, 2), "n_done": n_done, "n_total": len(work_cells)})

    t_bench0 = time.time()
    active_threads: list[threading.Thread] = []
    next_worker_id = 0
    while not work_q.empty() or any(t.is_alive() for t in active_threads):
        active_threads = [t for t in active_threads if t.is_alive()]

        if work_q.empty():
            # Nothing left to dispatch (workers only get_nowait, never put) --
            # just wait out the running cells. No more reprobing, and no
            # trailing sleep once they finish.
            for t in active_threads:
                t.join()
            break

        cap = probe_host(host_id, patch_km=patch_km)
        while len(active_threads) < cap.max_concurrency and not work_q.empty():
            t = threading.Thread(target=worker, args=(next_worker_id,), daemon=True)
            next_worker_id += 1
            t.start()
            active_threads.append(t)
        time.sleep(REPROBE_INTERVAL_S)
    t_bench1 = time.time()

    n_ok = sum(1 for r in results if not r.get("skipped") and r.get("returncode") == 0 and r.get("output_path"))

    # Every selected cell EXCEPT the cloud-skipped ones should have a raster;
    # _build_vrt globs by existence so a cloud skip is simply left out.
    cloud_skipped = {r["aoi_name"] for r in results if r.get("skip_reason") == "cloud"}
    done_aoi_names = [c["aoi_name"] for c in work_cells if c["aoi_name"] not in cloud_skipped]
    vrt_path = _build_vrt(date_str, vrt_subdir, done_aoi_names)

    return DailyRunResult(
        date_str=date_str, mode=mode, s1_metadata=s1_meta,
        n_selected=len(work_cells), n_ok=n_ok, n_skipped_already_done=n_skipped,
        total_wall_time_s=round(t_bench1 - t_bench0, 2),
        vrt_path=vrt_path, n_skipped_cloud=n_skipped_cloud, patches=results,
    )
