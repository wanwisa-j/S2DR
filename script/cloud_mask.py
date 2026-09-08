# S2_SuperResolution/utils/cloud_mask.py
#
# Post-process: null out cloud pixels in a delivered S2DR4 _TCI_1m.tif using
# that scene's OMNI_SCL_20m.tif (upsampled to the 1m grid, nearest-neighbor
# — categorical data). This is a mitigation applied AFTER inference — the
# S2DR4 docker image has no input-side mask parameter (confirmed by reading
# its own source: the real inference call `iu.test(point, date)` takes only
# a point and a date, nothing else), so this cannot stop the model from
# seeing cloud during generation, only keep hallucinated cloud texture out
# of the delivered raster.
#
# Processes in windows aligned to the TCI's own internal block tiling rather
# than loading full bands into memory — a whole-band version OOM-killed
# (exit 137) on a real ~68GB raster during dev testing (2026-08-11, tile
# 47PPS in the s2sr-corridor exploration). Memory use here is bounded by
# block size (typically 512x512), independent of raster size.
#
# Per-window cost: a real 10-patch/6km/GPU benchmark (2026-08-13, host 87)
# measured this step at 640-750s per cell, comparable to inference itself.
# The prior implementation called `rasterio.warp.reproject()` once per TCI
# block window (hundreds-thousands per cell) -- reproject() is a thin
# wrapper over GDAL's warp API, so switching library (rasterio -> raw GDAL)
# would not help; the cost is GDAL's per-call warp-context setup, repeated
# once per window regardless of which Python binding calls it. When SCL and
# TCI share a CRS (the common case -- both come from the same UTM zone; the
# rare exception is a cell whose mosaic spans a UTM boundary, reprojected to
# the majority zone by raster.py::mosaic), nearest-neighbor resampling
# reduces to pure pixel-index math: no coordinate transform, no GDAL warp
# context, just an affine map + array gather. That's what
# `_gather_same_crs` below does; reproject() stays as the fallback for the
# cross-CRS case and for any transform with a rotation term (unseen in
# practice for these products, but reproject() handles it correctly and the
# fast path's pure-affine math does not).
#
# geopandas/rasterio are imported lazily inside functions, matching the
# convention in corridor_grid_asset.py — repository.py imports every asset
# module just to register jobs/schedules, and that must stay possible
# without GDAL installed (e.g. CI's pytest job).

from __future__ import annotations

import glob

CLOUD_CLASSES = {1, 2}  # omnicloudmask scheme: 0=Clear,1=Thick,2=Thin,3=Shadow — NOT ESA SCL codes

# Same convention as the existing Sentinel-2 pipeline (see
# /fs2/thanakit/sentinel-2/s2_paths.py select_storage_root) — production S2
# tiles live under one of these roots depending on acquisition year.
_S2_ROOTS = ["/fs2/sentinel2", "/fs7/sentinel2"]


def find_omni_scl(mgrs_tile: str, date_str: str) -> str | None:
    """Locate the OMNI_SCL_20m.tif for (mgrs_tile, date_str) on the shared
    NAS. Returns None if the Sentinel-2 pipeline hasn't reached the
    omni_scl step for this tile/date yet (or the scene isn't there) —
    callers should treat that as "deliver unmasked", not an error."""
    zone, band, square = mgrs_tile[:2], mgrs_tile[2], mgrs_tile[3:]
    date_compact = date_str.replace("-", "")
    for root in _S2_ROOTS:
        pattern = (
            f"{root}/tiles/{zone}/{band}/{square}/S2*_MSIL2A_{date_compact}*_T{mgrs_tile}_*.SAFE/"
            f"GRANULE/*/IMG_DATA/R20m/T{mgrs_tile}_*_OMNI_SCL_20m.tif"
        )
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def _is_axis_aligned(transform) -> bool:
    """True if the transform has no rotation/shear terms (b == d == 0) --
    every product this pipeline produces is north-up, but this is cheap to
    verify rather than assume, and the fast path's math is only valid when
    it holds."""
    return transform.b == 0 and transform.d == 0


def _gather_same_crs(scl_arr, scl_transform, win_transform, win_h: int, win_w: int):
    """Nearest-neighbor resample of scl_arr onto a window's pixel grid via
    pure affine index math -- numerically equivalent to
    `rasterio.warp.reproject(..., resampling=Resampling.nearest)` when
    source and destination share a CRS and both transforms are axis-aligned
    (checked by the caller), but skips GDAL's per-call warp-context setup.

    Pixel-center convention matches GDAL: sample at each destination pixel's
    center, floor the source-space coordinate to get the covering source
    pixel (a pixel at index i covers source-space [i, i+1)). Destination
    pixels whose center falls outside the source array get 0 (Clear in the
    omnicloudmask scheme) -- the same value `reproject()` leaves untouched
    in a zero-initialized destination array for any area the source doesn't
    cover.
    """
    import numpy as np

    cols = np.arange(win_w)
    rows = np.arange(win_h)
    xs = win_transform.c + (cols + 0.5) * win_transform.a
    ys = win_transform.f + (rows + 0.5) * win_transform.e

    src_col = np.floor((xs - scl_transform.c) / scl_transform.a).astype(np.int64)
    src_row = np.floor((ys - scl_transform.f) / scl_transform.e).astype(np.int64)

    col_grid, row_grid = np.meshgrid(src_col, src_row)
    valid = (
        (col_grid >= 0)
        & (col_grid < scl_arr.shape[1])
        & (row_grid >= 0)
        & (row_grid < scl_arr.shape[0])
    )
    col_clipped = np.clip(col_grid, 0, scl_arr.shape[1] - 1)
    row_clipped = np.clip(row_grid, 0, scl_arr.shape[0] - 1)
    gathered = scl_arr[row_clipped, col_clipped]
    return np.where(valid, gathered, 0).astype(scl_arr.dtype)


class Scl:
    """One OMNI_SCL raster held in memory: read ONCE per (tile, date) and used
    for both the pre-dispatch cloud-fraction gate (cloud_fraction_over) and the
    post-inference mask (mask_cloud_nodata), so daily_run never re-reads it."""

    __slots__ = ("arr", "crs", "transform", "path")

    def __init__(self, arr, crs, transform, path):
        self.arr = arr
        self.crs = crs
        self.transform = transform
        self.path = path


def load_scl(omni_scl_path: str) -> Scl:
    """Read an OMNI_SCL_20m.tif into an Scl. Small raster (20m), whole-array."""
    import rasterio

    with rasterio.open(omni_scl_path) as ds:
        return Scl(ds.read(1), ds.crs, ds.transform, omni_scl_path)


def _as_scl(scl) -> Scl:
    return scl if isinstance(scl, Scl) else load_scl(scl)


def cloud_fraction_over(scl, geometry_wgs84) -> tuple[float, int]:
    """(cloud_fraction, n_pixels) for the SCL clipped to `geometry_wgs84`
    (a shapely polygon, WGS84 — the cell's true corridor shape, NOT its bbox).

    `cloud_fraction` counts the SAME classes mask_cloud_nodata nulls out
    ({1,2}); shadow is left out on purpose since it is delivered, not masked.
    `n_pixels` is the count inside the polygon — a small-sample guard for the
    caller, since a 150 m corridor strip is only ~7 SCL pixels wide.
    Returns (0.0, 0) when the polygon does not land on the SCL at all.
    """
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    scl = _as_scl(scl)
    shp = transform_geom("EPSG:4326", scl.crs, geometry_wgs84.__geo_interface__)
    inside = ~geometry_mask(
        [shp], out_shape=scl.arr.shape, transform=scl.transform, invert=False
    )
    n = int(inside.sum())
    if n == 0:
        return 0.0, 0
    cloud = np.isin(scl.arr, list(CLOUD_CLASSES)) & inside
    return int(cloud.sum()) / n, n


def mask_cloud_nodata(tci_path: str, scl) -> dict:
    """Null out cloud pixels ({1,2}) from `scl` in tci_path, in place,
    processed window-by-window. `scl` is an Scl from load_scl (preferred, no
    re-read) or a path to an OMNI_SCL_20m.tif. Returns a small stats dict for
    the calling asset's materialization metadata."""
    from datetime import datetime, timezone

    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import transform as window_transform

    start = datetime.now(timezone.utc)

    scl = _as_scl(scl)
    scl_arr = scl.arr
    scl_crs = scl.crs
    scl_transform = scl.transform
    omni_scl_path = scl.path

    n_cloud_px = 0
    n_total_px = 0

    with rasterio.open(tci_path, "r+") as tci_ds:
        nodata = tci_ds.nodata if tci_ds.nodata is not None else 0
        tci_crs = tci_ds.crs
        windows = [w for _, w in tci_ds.block_windows(1)]

        same_crs = scl_crs == tci_crs and _is_axis_aligned(scl_transform)

        for win in windows:
            win_transform = window_transform(win, tci_ds.transform)
            win_h, win_w = int(win.height), int(win.width)

            if same_crs and _is_axis_aligned(win_transform):
                scl_on_win = _gather_same_crs(scl_arr, scl_transform, win_transform, win_h, win_w)
            else:
                scl_on_win = np.zeros((win_h, win_w), dtype=scl_arr.dtype)
                reproject(
                    source=scl_arr,
                    destination=scl_on_win,
                    src_transform=scl_transform,
                    src_crs=scl_crs,
                    dst_transform=win_transform,
                    dst_crs=tci_crs,
                    resampling=Resampling.nearest,
                )
            cloud_mask = np.isin(scl_on_win, list(CLOUD_CLASSES))
            n_total_px += cloud_mask.size
            if not cloud_mask.any():
                continue

            for band_idx in range(1, tci_ds.count + 1):
                band = tci_ds.read(band_idx, window=win)
                band[cloud_mask] = nodata
                tci_ds.write(band, band_idx, window=win)
            n_cloud_px += int(cloud_mask.sum())

    end = datetime.now(timezone.utc)
    return dict(
        duration_s=round((end - start).total_seconds(), 2),
        n_cloud_px_masked=n_cloud_px,
        n_total_px=n_total_px,
        cloud_pct_of_raster=round(100 * n_cloud_px / n_total_px, 2) if n_total_px else None,
        omni_scl_path=omni_scl_path,
    )
