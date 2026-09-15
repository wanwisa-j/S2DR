# S2_SuperResolution/utils/core_runner.py
#
# Remote-execution shim: Dagster -> SSH -> a target server -> docker run.
# All Docker/GPU/CPU work happens on whichever server is named by host_id
# (image already pulled on every host we dispatch to). This file only builds
# the `docker run ... box --aoi-file --date` command and issues it over SSH —
# it never touches the ei-geoint-superresolution image or its internal
# logic, which is owned/maintained outside this repo.
#
# host_id-agnostic by design (see capacity_probe.py, which decides WHICH
# host + patch_km/concurrency to use before calling into this module) —
# every SSH call takes host_id explicitly rather than hardcoding server 95,
# so adding a second host is a .env change, not a code change.
#
# Required .env keys per host (server-based naming, see docker-compose SSH
# block) — <N> is the host_id passed to every function here, e.g. "95":
#   SSH_HOST_<N>   IP of the server        (e.g. 172.23.68.95)
#   SSH_USER_<N>   login user on that server (e.g. wanwisaj)
#   SSH_KEY_<N>    path to private key
#   SSH_PORT_<N>   ssh port                (default 22)

import os
import subprocess
from pathlib import Path

IMAGE = "registry.earthinsights.space/earthinsights/ei-geoint-superresolution:0.1.0"

# Staging stays on /fs2 -- pure scratch space, cleared per-run, no reason
# to move it. Final output moved to /fs8 2026-08-20 (bigger, shared-project
# volume -- see remote_date_dir) so downloaded mosaics land where the rest
# of the team's Deep_resolution data already lives, not buried in a
# personal /fs2/wanwisa dev tree. Confirmed /fs8 is mounted identically on
# every dispatch host (checked host 87 directly, same NFS export as /fs2),
# so nothing else about the shared-mount/de-dup reasoning below changes.
REMOTE_OUT_BASE = "/fs8/project_data/tso_data/Shared_data/Deep_resolution/Daily"
# On /fs8, not /fs2: this scratch dir used to live on /fs2 and a 2026-09-12/15
# NAS outage on 172.16.99.203 (serves /fs2 + /fs7) proved that a hung `mkdir -p`
# on ANY operand blocks the whole command -- since dispatch does
# `mkdir -p {out_dir} {stage_dir} && docker run ...` in one shot, a stuck
# staging dir freezes the cell even though the real output (/fs8, a different
# NAS, 172.16.99.204) was healthy the entire time. /fs8 confirmed mounted
# identically on every dispatch host (see REMOTE_OUT_BASE above).
REMOTE_STAGE_ROOT = "/fs8/wanwisa/s2dr-staging"

# --- on-prem Sentinel-2 source, opt-in (thcom-s2dr4 THCOM_LOCAL_SOURCE) -------
# The container reads source bands from these NAS roots instead of
# sentinel-cogs S3 where the mirror has the scene (see thcom-s2dr4
# src/core/local_source.py + docs/local-source.md). 5-patch parity on host 87
# (2026-09-07): ~4% faster wall, TCI matches S3 within rounding on 4/5 cells
# (one cell drifted 2.2% of pixels / max 50 DN from co-registration
# nondeterminism between JP2 and COG inputs -- accepted). Real value is
# resilience when S3 is throttling/down, not the speed.
#
# OFF by default: only turn on AFTER the ei-geoint-superresolution image is
# rebuilt from a thcom-s2dr4 build that carries local_source.py + the
# use_local_source lever (an older image just ignores THCOM_LOCAL_SOURCE and
# stays on S3 -- inert, not broken). Enable with S2DR_USE_LOCAL_SOURCE=1 in the
# orchestrator .env, then run one manual s2dr cell and check its output before
# trusting a nightly.
#
# The BOA-offset VRTs the container builds are cached here, shared across every
# cell and every run (NFS, ~1 KB each) so the one-time ~90 s first-cell build
# is paid once ever, not per cell.
USE_LOCAL_SOURCE = os.environ.get("S2DR_USE_LOCAL_SOURCE", "0") == "1"
REMOTE_S2_MIRROR_ROOTS = ("/fs2/sentinel2", "/fs7/sentinel2")
REMOTE_VRT_CACHE = "/fs2/wanwisa/s2sr-corridor/vrtcache"


def remote_date_dir(date_str: str) -> str:
    """Public accessor -- per-date root, e.g.
    /fs8/.../Deep_resolution/Daily/2026/08/2026-08-03/. Parent of every
    (date, cell) output dir below AND of the full/priority VRT subdirs
    daily_run.py builds (see its _build_vrt) -- one function so both stay
    in sync instead of each hardcoding the yyyy/mm nesting separately."""
    yyyy, mm = date_str[:4], date_str[5:7]
    return f"{REMOTE_OUT_BASE}/{yyyy}/{mm}/{date_str}"


def _remote_out_dir(date_str: str, aoi_name: str) -> str:
    """Per-(date, cell) output dir -- NOT per-date-only, same reasoning as
    _remote_stage_dir below. This host path is bind-mounted to the
    container's fixed internal /opt/ml/processing/output, and the runbook's
    own documented layout (run_book/s2_super_resolution.md, pre-fix) shows
    it holds more than just the final mosaic: a `TH/` subfolder of raw
    per-patch rasters BEFORE mosaicking, shared by every cell processed on
    that date. A real 2-cell CPU test (host 95, 2026-08-13) on two different
    aoi_names produced byte-identical mosaic_input_count (34) and identical
    output size on both cells -- only explainable by one cell's container
    seeing the other's leftover TH/ patches through this shared mount. A
    controlled clear-then-rerun (2026-08-14, host 87, orbit99 cells) then
    confirmed causation directly: OOM with a stale/shared output dir,
    clean success at the same patch_km/date immediately after clearing it.
    Keying by (date_str, aoi_name) isolates each cell's container output
    the same way the staging fix already isolates its scratch space --
    without this, a daily schedule would re-accumulate the same bloat
    across cells within a single day and OOM again regardless of the
    staging fix."""
    return f"{remote_date_dir(date_str)}/{aoi_name}"


def _remote_stage_dir(date_str: str, aoi_name: str) -> str:
    """Per-(date, cell) staging dir -- NOT per-date-only. This host path is
    bind-mounted to the container's fixed internal /content/output, which
    the compiled library uses as its own scratch/resume space with no
    AOI-awareness. A shared per-date-only directory let one cell's leftover
    scratch files be visible to every later cell's container on the same
    date -- confirmed by a real 10-patch benchmark (2026-08-13, host 87):
    mosaic_input_count grew monotonically across the batch (16->30) despite
    every cell having only 1-2 grid points of its own, and one cell's mosaic
    pulled in 27 other cells' leftovers, producing a 1.4GB intermediate
    raster and exceeding the memory cap. Keying by (date_str, aoi_name)
    isolates each cell's container to its own scratch space."""
    return f"{REMOTE_STAGE_ROOT}/{date_str}/{aoi_name}"


# ---------------------------------------------------------------------------
# SSH helpers (same shape as Durian_NDVI / Image_alignment core_runner.py),
# now parameterized by host_id instead of hardcoding server 95.
# ---------------------------------------------------------------------------


def _ssh_argv(host_id: str) -> list[str]:
    host = os.environ.get(f"SSH_HOST_{host_id}", "").strip()
    if not host:
        raise RuntimeError(
            f"SSH_HOST_{host_id} is not set. Add it to .env: SSH_HOST_{host_id}=<ip>"
        )
    user = os.environ.get(f"SSH_USER_{host_id}", "").strip()
    key = os.environ.get(f"SSH_KEY_{host_id}", "").strip()
    port = os.environ.get(f"SSH_PORT_{host_id}", "22").strip()

    target = f"{user}@{host}" if user else host
    flags = [
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=30",
    ]
    if key:
        flags += ["-i", key]
    return ["ssh", *flags, target]


def _ssh_run(host_id: str, remote_bash: str, *, timeout: int = 3600) -> subprocess.CompletedProcess:
    cmd = [*_ssh_argv(host_id), remote_bash]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _raise_on_failure(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode == 0:
        return
    stderr_tail = (result.stderr or "")[-4000:]
    stdout_tail = (result.stdout or "")[-2000:]
    raise RuntimeError(
        f"{label} exited with code {result.returncode}\n"
        f"--- STDERR ---\n{stderr_tail}\n"
        f"--- STDOUT (tail) ---\n{stdout_tail}"
    )


# ---------------------------------------------------------------------------
# Dedup / output validation (read via NFS — /fs2 is mounted on every host,
# including wherever this Dagster code runs, so no SSH needed here)
# ---------------------------------------------------------------------------


def output_already_done(aoi_name: str, date_str: str) -> bool:
    """True if S2DR already produced this (aoi_name, date), on ANY host.

    NOTE: the `.S2SR10x_{AOI}_{DATE}.done` file is NOT a completion marker —
    it's the container's internal per-patch resume tracker and can exist
    (partially populated) long before the job actually finishes. Checking
    it here would make dedup falsely skip AOIs that never finished. The
    only reliable signal is the final mosaic GeoTIFF actually existing.

    Also true (via 2026-08-11 dev testing, see plan) when the container was
    killed (exit 137, almost always OOM) AFTER writing this file but before
    writing `_metadata.json` — the mosaic itself is real and complete in
    that case, so checking file existence rather than trusting the exit
    code is deliberate, not just a de-dup convenience.
    """
    date_compact = date_str.replace("-", "")
    tif = Path(_remote_out_dir(date_str, aoi_name)) / f"S2SR10x_{aoi_name}_{date_compact}_TCI_1m.tif"
    return tif.exists()


def validate_output(aoi_name: str, date_str: str) -> Path:
    """Assert the super-resolved GeoTIFF exists after a run. Checked by file
    existence, not the docker process's exit code — see output_already_done
    for why exit code alone is unreliable for this image."""
    date_compact = date_str.replace("-", "")
    tif = Path(_remote_out_dir(date_str, aoi_name)) / f"S2SR10x_{aoi_name}_{date_compact}_TCI_1m.tif"
    if not tif.exists():
        raise RuntimeError(
            f"S2DR validation FAILED: output file not found at {tif}. "
            "Container exited without writing the expected output file."
        )
    return tif


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_s2_super_resolution(
    host_id: str,
    aoi_file: str,
    date_str: str,
    aoi_name: str,
    patch_km: float,
    gpu_available: bool,
    threads: int = 4,
) -> subprocess.CompletedProcess:
    """
    Execute `docker run ... box --aoi-file <geojson> --date D` on `host_id`
    via SSH.

    `aoi_file` is a per-cell GeoJSON polygon (cell square clipped to the
    real 150m corridor buffer — see corridor_grid_asset.py), not a bbox:
    `--aoi-file` mode clips the output to the true polygon shape (confirmed
    via the vendor's own --help text and direct dev testing) — a plain bbox
    would deliver a rectangle that still includes real, non-corridor
    imagery just outside the 150m buffer.

    `gpu_available` and `threads` come from capacity_probe.probe_host() —
    this function only builds/dispatches the command for whatever decision
    was already made; it does not re-check capacity itself. `threads` MUST
    be the real per-worker core budget computed against the TARGET host's
    own core count (capacity_probe queries `nproc` over SSH on that host) —
    never `os.cpu_count()` here, which would read the Dagster container's
    own host, not the remote one. CPU mode always forces
    CUDA_VISIBLE_DEVICES="" and caps every thread-pool env var to `threads`
    so sibling containers on the same host don't oversubscribe cores (the
    load-average-37 lockup incident this guards against is documented in
    s2dr4-run's dev_tests/parallel_concurrency_v1).

    Only constructs/dispatches the command — the container's internal
    behavior (fetching Sentinel-2 source imagery, running S2DR4, mosaicking)
    is entirely out of scope here.
    """
    mem_gb = _memory_limit_gb(patch_km, gpu_available)
    out_dir = _remote_out_dir(date_str, aoi_name)
    stage_dir = _remote_stage_dir(date_str, aoi_name)
    aoi_dir = str(Path(aoi_file).parent)
    aoi_filename = Path(aoi_file).name

    gpu_flags = ""
    env_flags = f"-e THCOM_PATCH_KM={patch_km} -e THCOM_AOI_NAME={aoi_name} "
    if gpu_available:
        gpu_flags = "--gpus all "
    else:
        env_flags += (
            '-e CUDA_VISIBLE_DEVICES="" '
            f"-e OMP_NUM_THREADS={threads} -e MKL_NUM_THREADS={threads} "
            f"-e OPENBLAS_NUM_THREADS={threads} -e NUMEXPR_NUM_THREADS={threads} "
        )

    # Mount only the mirror roots that actually exist on THIS dispatch host
    # (built in the remote shell so a host without /fs7 still works), plus the
    # shared VRT cache. See USE_LOCAL_SOURCE note above.
    local_src_setup = ""
    local_src_flags = ""
    if USE_LOCAL_SOURCE:
        roots = " ".join(REMOTE_S2_MIRROR_ROOTS)
        local_src_setup = (
            f"mkdir -p {REMOTE_VRT_CACHE} && MIRROR_MOUNTS='' && "
            f'for r in {roots}; do if [ -d "$r" ]; then '
            f'MIRROR_MOUNTS="$MIRROR_MOUNTS -v $r:$r:ro"; fi; done && '
        )
        local_src_flags = (
            f"$MIRROR_MOUNTS -v {REMOTE_VRT_CACHE}:/vrtcache "
            f"-e THCOM_LOCAL_SOURCE=1 -e LOCAL_SOURCE_VRT_DIR=/vrtcache "
        )

    remote_cmd = (
        f"mkdir -p {out_dir} {stage_dir} && "
        f"{local_src_setup}"
        f"docker run --rm {gpu_flags}"
        f'--user "$(id -u):$(id -g)" '
        f"--memory {mem_gb}g "
        f"-v {out_dir}:/opt/ml/processing/output "
        f"-v {stage_dir}:/content/output "
        f"-v {aoi_dir}:/content/aoi:ro "
        f"{local_src_flags}"
        f"{env_flags}"
        f"{IMAGE} -m src.cli box "
        f"--aoi-file /content/aoi/{aoi_filename} "
        f"--date {date_str}"
    )
    result = _ssh_run(host_id, remote_cmd, timeout=3600 * 6)
    _raise_on_failure(result, f"S2DR docker run (host={host_id}, aoi={aoi_name}, date={date_str})")
    return result


def _memory_limit_gb(patch_km: float, gpu_available: bool) -> int:
    """--memory cap for docker, sized from patch_km with margin. GPU mode
    still needs a system-RAM cap even though the model itself runs on the
    GPU (mosaic/clip and scene I/O are CPU-side) — kept modest since GPU
    mode is currently capped at concurrency=1 anyway.

    Was 24g: a real 10-patch/6km/GPU benchmark (host 87, 2026-08-13) hit
    exit 137 (OOM) on 3/3 completed patches, every time after mosaic and
    before the container's own clip/metadata step. Exact peak isn't known
    (only system-wide RAM was sampled, not this container's own cgroup),
    so 48g is a 2x-headroom guess against the 24g that failed, not a
    measured requirement — revisit once a run is observed to actually
    need less, or still OOMs, at this cap."""
    if gpu_available:
        return 48
    # Lazy on purpose (capacity_probe imports core_runner at its own top
    # level -- promoting this to module level would be circular). The
    # sys.path re-insert guards against the caller's sys.path.insert() for
    # this directory not being visible here anymore by the time this lazy
    # import actually runs (Dagster's loader restores sys.path after the
    # initial module-load window that any top-level sibling import relies
    # on -- see S2DR/assets/s2dr_daily_run_asset.py in the git repo).
    import sys as _sys

    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from capacity_probe import CPU_MEM_GB_PER_PATCH_KM

    base = CPU_MEM_GB_PER_PATCH_KM.get(patch_km, 40.0)
    return (
        int(base * 1.4) + 5
    )  # matches the real-world margin s2dr4-run needed (20-23GB -> 40g cap)
