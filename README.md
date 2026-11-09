# s2sr-corridor

Daily dispatcher for the Sentinel-2 10x super-resolution (S2DR4) job. This
repo does **not** contain the super-resolution model itself — it SSHes into
a GPU host and runs a pre-built vendor docker image (see
[`script/core_runner.py`](script/core_runner.py)) once per AOI cell per day,
then mosaics the results.

Called from Dagster (`dagster/S2DR/assets/s2dr_daily_run_asset.py` in the
orchestrator repo) via:
```python
sys.path.insert(0, "/fs2/wanwisa/s2sr-corridor/script")
from daily_run import run_daily
run_daily(date_str, mode)  # mode: "priority" or "full"
```
`run_daily` is framework-agnostic — it also runs standalone from a plain
Python shell (see Quickstart below), which is how new setups should be
tested before wiring them into Dagster.

## Architecture

```
run_daily(date_str, mode)
  -> select_covered_cells()   # which grid cells have real S2 coverage today
  -> s1_check.check()         # optional SAR cross-check
  -> per cell, per worker (concurrency from capacity_probe.probe_host):
       core_runner: SSH -> host -> `docker run <IMAGE> box --aoi-file ... --date ...`
  -> generate_preview.generate_preview()  # full/priority VRT + PNG for the day
```

- `script/capacity_probe.py` — asks a host live (GPU usable? free RAM?) how
  many workers it can safely run right now; host-agnostic, add a host via
  `.env`, not code.
- `script/core_runner.py` — builds and issues the `docker run` command over
  SSH. Owns every hardcoded path below.
- `script/cloud_mask.py`, `script/stac_check.py`, `script/tile_grouping.py`,
  `script/s1_check.py` — scene selection / cloud-cover / SAR cross-check
  inputs to `daily_run.py`.
- `script/retention.py` — deletes output/staging date-folders older than
  `retention_days` (default 14). **Not currently scheduled anywhere** — run
  it by hand (`plan_cleanup()` first, always; `cleanup(dry_run=False)` only
  after reviewing the plan) if disk needs reclaiming. Nothing deletes old
  data automatically right now.

## ⚠️ Before you run this yourself — config that is hardcoded to THIS deployment

Every one of these lives in `script/core_runner.py` or `script/daily_run.py`
and points at a specific team's infrastructure. Cloning this repo and
running it unchanged will try to write into **our** production data and SSH
into **our** GPU host. Change all of these first:

| Constant | File | Current value | Why you must change it |
|---|---|---|---|
| `IMAGE` | `core_runner.py` | `registry.earthinsights.space/earthinsights/ei-geoint-superresolution:0.1.0` | Points at our private registry. You need your own S2DR4 image (see [thcom-s2dr4](https://github.com/thcom-anawat/thcom-s2dr4) to build one) and your own tag here. |
| `REMOTE_OUT_BASE` | `core_runner.py` | `/fs8/project_data/tso_data/Shared_data/Deep_resolution/Daily` | **This is our team's shared production data folder.** Running unchanged will write your test output into it. Point this at your own path. |
| `REMOTE_STAGE_ROOT` | `core_runner.py` | `/fs8/wanwisa/s2dr-staging` | Scratch space, but still a path under a specific user's home — point at your own. |
| `REMOTE_VRT_CACHE` | `core_runner.py` | `/fs2/wanwisa/s2sr-corridor/vrtcache` | Only matters if `S2DR_USE_LOCAL_SOURCE=1` (off by default). |
| `GRID_PATH` | `daily_run.py` | `grid-7km5/corridor_grid.parquet` | This is **our** AOI (a specific pipeline corridor), not yours. Build your own grid — see below. |
| `HOST_ID` | `daily_run.py` | `"87"` | Our GPU host's `.env` key suffix. You need your own host + matching `.env` entries (below). |

None of these are read from `.env` or a config file — they're module-level
constants. Edit them directly, or fork the constants into your own config
module if you want to keep upstream changes mergeable.

### `.env` — SSH host config (per host, `<N>` = the `host_id` you pick)

```
SSH_HOST_<N>=<ip>
SSH_USER_<N>=<login user>
SSH_KEY_<N>=<path to private key>
SSH_PORT_<N>=22            # optional, defaults to 22
```
Host-agnostic by design — adding a second host is a `.env` change, never a
code change. `HOST_ID` in `daily_run.py` just has to match one of your `<N>`s.

### Building your own grid

`grid-7km5/corridor_grid.parquet` is a plain DataFrame, one row per AOI cell:

```
cell_id, bbox_west, bbox_south, bbox_east, bbox_north, patch_km, aoi_name, aoi_file
```

`aoi_file` points at a per-cell GeoJSON (WGS84 — reproject before writing;
the vendor image expects WGS84 regardless of your source CRS) under
`<grid-dir>/aoi_cells/<aoi_name>.geojson`. There's no grid-builder script in
this repo (ours was built once, ad hoc) — the pattern is: pick a cell size
**smaller than your `patch_km`** (we use 7.5km cells with an 8km patch,
specifically so every cell needs exactly 1 patch — see the "minimum unit
cost = 1 patch" note below), tile your AOI polygon into cells of that size,
keep only cells that intersect it, write each as its own GeoJSON, and
collect the columns above into a parquet. Point `GRID_PATH` at the result.

## Quickstart — test before you schedule anything

**Always smoke-test one cell manually before running a full day.** The
vendor model's real minimum processing unit is one full patch (`patch_km`
wide — 8km × 8km = 64 km² by default), regardless of how small your actual
AOI cell is: a 1 km² AOI still costs the same GPU time, RAM reservation, and
`TH/`-folder intermediate storage as a full patch, because the model always
infers the whole patch and only *crops* the output to your AOI afterward.
Budget and time estimates should be done per-patch, never by linearly
scaling a big-patch cost down to a smaller nominal area.

1. Set up `.env` (above) and confirm SSH + docker work on your host:
   ```
   ssh -i $SSH_KEY_<N> $SSH_USER_<N>@$SSH_HOST_<N> docker run --rm <your image> -m src.cli box --help
   ```
2. Build (or copy in) a small grid — even 1 cell is enough for this step.
3. From a Python shell in `script/`, run one cell directly through
   `core_runner` (bypass `daily_run.py`'s concurrency/scheduling entirely)
   and check the result before trusting a scheduled run:
   ```python
   import core_runner as r
   from capacity_probe import probe_host
   cap = probe_host("<N>", patch_km=8.0)
   result = r.run_s2_super_resolution(
       host_id="<N>", aoi_file="/path/to/test.geojson", date_str="YYYY-MM-DD",
       aoi_name="TEST", patch_km=8.0, gpu_available=cap.gpu_available,
       threads=cap.threads_per_worker,
   )
   r.validate_output(aoi_name="TEST", date_str="YYYY-MM-DD")  # raises if the mosaic TIF is missing
   ```
   Confirm the final mosaic TIF exists under your `REMOTE_OUT_BASE`, and
   check the requested date actually has an exact Sentinel-2 scene (the
   pipeline logs `requested date NOT available` + substitutes the nearest
   scene if not — decide up front whether that's acceptable for your use
   case, since `--no-exact-date` does **not** make the mosaic step use the
   substituted date).
4. Only after a clean single-cell result, call `run_daily(date_str, mode,
   host_id="<N>")` for a full day.

## Storage note

Each cell's output dir holds the final mosaic TIF/metadata/footprint *plus*
a `TH/` subfolder of raw per-patch rasters from before mosaicking — roughly
the same size as the final TIF, sometimes larger. `TH/` isn't pure scratch:
`metadata.json`'s `sources[].file` references it, and a re-run of the same
cell+date skips already-done patches by checking for their file there
(`patches_already_done` in the metadata). Nothing currently prunes it
automatically except the unscheduled `retention.py` (14-day window, deletes
the whole date-folder, `TH/` included).
