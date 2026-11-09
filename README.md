# s2sr-corridor

Daily dispatcher for the Sentinel-2 10x super-resolution (S2DR4) job. This
repo does **not** contain the super-resolution model itself and does **not**
build any docker image — it SSHes into a GPU host and runs an
**already-built** image once per AOI cell per day, then mosaics the results.

**Where the image comes from:** this team's image
(`registry.earthinsights.space/earthinsights/ei-geoint-superresolution:0.1.0`,
pinned in `script/core_runner.py`'s `IMAGE` constant) is built from
[thcom-s2dr4](https://github.com/thcom-anawat/thcom-s2dr4) — a *separate*
repo that owns the model, the Dockerfile, and the vendor weights, and has no
shared git history with this one. If you don't already have your own image
built from `thcom-s2dr4` (or equivalent), **you cannot run this repo** —
there is nothing here that builds one for you.

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
- `script/retention.py` — **exists but nothing calls it.** Right now,
  production keeps every date-folder forever — confirmed directly: output
  going back to 2025-03 is all still there, nothing has ever been
  auto-deleted. `retention.py` is a manual tool you can run by hand
  (`plan_cleanup()` first, always, to see what it *would* delete;
  `cleanup(dry_run=False)` only after reviewing that plan) if disk ever
  needs reclaiming — it is not wired into any schedule or cron, so until
  someone runs it, storage only grows.

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

`grid-7km5/corridor_grid.parquet` (production's grid) is a plain DataFrame,
one row per AOI cell:

```
cell_id, bbox_west, bbox_south, bbox_east, bbox_north, patch_km, aoi_name, aoi_file
```

`aoi_file` points at a per-cell GeoJSON (WGS84 — reproject before writing;
the vendor image expects WGS84 regardless of your source CRS).

`script/build_grid.py` builds this from any AOI boundary file geopandas can
read (Shapefile, GeoPackage, GeoJSON):

```
python script/build_grid.py your_aoi.shp ./my_grid --prefix MYPROJECT
```

Tiles your AOI into cells **smaller than `--patch-km`** (default 7.5km
cells / 8km patch — the same margin production uses, so every cell needs
exactly 1 patch; see the "minimum unit cost = 1 patch" note below), keeps
only cells that actually intersect the AOI polygon, and writes
`./my_grid/grid.parquet` + `./my_grid/aoi_cells/MYPROJECT-NNNN.geojson`.
This is separate from (and doesn't touch) `grid-7km5/`, production's own
grid.

### Running your own grid

`script/run_adhoc.py` dispatches a grid built above — resumable, retries a
transient failure once, concurrency-aware (via `capacity_probe`) — without
touching `daily_run.py` or its production wiring at all:

```
python script/run_adhoc.py ./my_grid/grid.parquet \
  --dates 2026-01-10 2026-01-25 \
  --host-id <N> --image <your image> \
  --out-base /your/own/output/path --stage-base /your/own/staging/path
```

`--host-id`, `--image`, `--out-base` and `--stage-base` are all required —
deliberately no defaults, so this can never silently fall back to this
team's registry image, production output path, or GPU host.

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
4. Only after a clean single-cell result, move to `script/run_adhoc.py`
   (above) for a full batch — or `run_daily(date_str, mode, host_id="<N>")`
   if you're wiring this into your own Dagster deployment.

## Storage note

**Nothing is ever deleted automatically.** Every date-folder this has ever
produced is still there — see `retention.py` above if you need to reclaim
space; until you run it by hand, storage only grows.

Each cell's output dir holds the final mosaic TIF/metadata/footprint *plus*
a `TH/` subfolder of raw per-patch rasters from before mosaicking — roughly
the same size as the final TIF, sometimes larger. `TH/` isn't pure scratch:
`metadata.json`'s `sources[].file` references it, and a re-run of the same
cell+date skips already-done patches by checking for their file there
(`patches_already_done` in the metadata) — so deleting it trades disk space
for losing that resume capability, if you ever decide to clean it up.
