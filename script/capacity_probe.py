# S2_SuperResolution/utils/capacity_probe.py
#
# "Auto worker" — before dispatching a docker run to a given host, ask that
# host what it can actually handle RIGHT NOW: is its GPU usable by this
# specific image, and given free memory, what --patch-km / how many
# concurrent workers is safe.
#
# GPU usability is checked by actually running `torch.cuda.is_available()`
# inside the real vendor image on that host — not by parsing driver version
# strings. Confirmed on server 95 (2026-08-11): driver 545.29.06 reports CUDA
# 12.3, but this image's torch build is 2.13.0+cu130 (needs CUDA 13.0) ->
# `cuda available: False` every time, independent of any docker flag. A
# version-string heuristic would have to hardcode that exact compatibility
# boundary and get it wrong the next time either side changes; asking torch
# directly is authoritative and self-updating.
#
# CPU-mode memory-per-worker table is empirical (see docstring on
# CPU_MEM_GB_PER_PATCH_KM). GPU VRAM-per-worker is NOT yet measured — no run
# has ever completed in GPU mode (GPU has never been usable so far). GPU mode
# is deliberately conservative (patch_km=6, concurrency=1) until a real GPU
# run is observed and this gets calibrated — see the plan's Verification
# section.

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from core_runner import IMAGE, _ssh_run

# Empirical, CPU-only mode. 8km confirmed directly (s2dr4-run project,
# CLAUDE.md: "8km ~20-23GB"). 12km confirmed directly from the vendor's own
# `--help` text ("12 km needs ~70 GB"). 6km is interpolated (patch-km^2
# scaling) from those two anchors, not independently measured yet.
CPU_MEM_GB_PER_PATCH_KM: dict[float, float] = {
    6.0: 15.0,
    8.0: 23.0,
    12.0: 70.0,
}

# GPU RAM-per-worker, empirical (host 87, RTX A6000, 2026-08-14/17 clean
# runs, aoi_name-scoped output+staging so no cross-cell contamination):
#   6km:  peak 26.4GB (2 clean runs)         -> +buffer for small sample size
#   8km:  peak 26.4GB (2 clean runs, ~same as 6km -- doesn't scale linearly
#         with area the way CPU mode's table does)
#   12km: peak 48.06GB (right at the old 48g cap -- see core_runner's
#         _memory_limit_gb docstring)         -> +2GB buffer, same reasoning
# Unlike CPU_MEM_GB_PER_PATCH_KM, this is NOT extrapolated for other
# patch_km values (12km's jump from 6/8km was not a smooth function of
# area) -- only these three measured sizes are trusted; anything else
# falls back to the 12km (largest measured) figure as the conservative
# default.
GPU_MEM_GB_PER_PATCH_KM: dict[float, float] = {
    6.0: 30.0,
    8.0: 30.0,
    12.0: 50.0,
}

# VRAM-per-worker, empirical: peak observed was a flat ~9.2GB regardless of
# patch_km (6/8/12km all landed at the same ~9208 MiB) -- padded with
# margin. Host 87's A6000 has 49GB total, so this was never the binding
# constraint in practice; RAM is (see module docstring above).
GPU_VRAM_GB_PER_WORKER = 12.0

# Same spirit as CPU mode's SAFETY_FACTOR, but slightly less conservative
# (0.75 vs 0.7) since GPU-mode capacity is now checked live right before
# each cell's own dispatch (not once at 03:00 for the whole day — see
# s2dr_availability_schedule.py), so the data behind this number is fresher
# than a once-daily snapshot would be. Deliberately NOT 0.9+: concurrent
# workers all read this same live snapshot before any of them have actually
# claimed memory, so a thin margin is a real race-condition risk, not just
# theoretical (see 2026-08-14 discussion — 90%+ was proposed and rejected
# for this reason).
GPU_SAFETY_FACTOR = 0.75

# Reserve at least this many CPU threads per worker (24 cores / N workers
# must stay usable) — co-registration/mosaic is CPU-bound (py-spy profile,
# 2026-08-14: GPU forward-pass itself was only ~0.84% self-time), so
# concurrency isn't free the way it would be for a purely GPU-bound job;
# this just stops concurrency from being computed so high that each worker
# gets starved down to a single core.
GPU_MIN_THREADS_PER_WORKER = 2

GPU_PATCH_KM = 6.0  # default patch_km when the caller doesn't specify one

# Fraction of free memory we're willing to actually use — leaves headroom for
# the host's own other work (this is a shared machine; see run_book) and for
# the fact that "free" at probe time can drop before the container starts.
SAFETY_FACTOR = 0.7

# Cache a host's GPU-usability verdict for this long — it's a fixed property
# of (image, driver) that won't change between two dispatches a few minutes
# apart, and the check itself pulls/starts a container so isn't free.
_GPU_CHECK_CACHE_TTL_S = 600
_gpu_check_cache: dict[str, tuple[float, bool]] = {}


@dataclass
class HostCapacity:
    host_id: str
    reachable: bool
    gpu_available: bool
    patch_km: float
    max_concurrency: int
    free_mem_mb: int
    checked_at: float
    cpu_count: int = 0
    error: str | None = None

    @property
    def score(self) -> float:
        """Relative weight for splitting work across hosts — concurrency is
        the throughput-limiting factor (each worker processes one cell at a
        time), so it's the natural weight."""
        return float(self.max_concurrency) if self.reachable else 0.0

    @property
    def threads_per_worker(self) -> int:
        """Real per-worker core budget on THIS host, for
        core_runner.run_s2_super_resolution's `threads` argument — must
        come from here (queried via SSH against this exact host), never
        from the Dagster container's own os.cpu_count()."""
        if self.max_concurrency < 1 or self.cpu_count < 1:
            return 1
        return max(1, self.cpu_count // self.max_concurrency)


def _host_env(host_id: str, key: str, default: str = "") -> str:
    return os.environ.get(f"{key}_{host_id}", default).strip()


def host_configured(host_id: str) -> bool:
    """True if this host has SSH connection details in .env at all — lets
    callers skip a host (e.g. 87, before its access is set up) without
    erroring the whole schedule."""
    return bool(_host_env(host_id, "SSH_HOST"))


def _check_gpu_usable(host_id: str) -> bool:
    now = time.time()
    cached = _gpu_check_cache.get(host_id)
    if cached and (now - cached[0]) < _GPU_CHECK_CACHE_TTL_S:
        return cached[1]

    # IMAGE's own ENTRYPOINT is ["python3"] (confirmed against the vendor's
    # image directly) -- args after IMAGE become CMD, appended to that
    # ENTRYPOINT. A leading "python3" here used to double up into
    # `python3 python3 -c ...`, which always failed (Python tried to run a
    # script literally named "python3") and made this probe report
    # gpu_available=False unconditionally, regardless of real GPU status
    # (confirmed 2026-08-13 via a standalone corrected probe; not fixed
    # here at the time since this file was out of scope for that
    # investigation). Just `-c ...` lets the image's own entrypoint supply
    # python3.
    probe_cmd = (
        f"docker run --rm --gpus all {IMAGE} -c "
        "'import torch; print(\"CUDA_OK\" if torch.cuda.is_available() else \"CUDA_NO\")' "
        "2>/dev/null | tail -1"
    )
    try:
        result = _ssh_run(host_id, probe_cmd, timeout=120)
        usable = result.returncode == 0 and "CUDA_OK" in (result.stdout or "")
    except Exception:
        # No GPU on the host at all (e.g. `--gpus all` itself fails), or SSH
        # hiccup — treat as "no GPU", not as a probe failure; CPU mode always
        # has a fallback path.
        usable = False

    _gpu_check_cache[host_id] = (now, usable)
    return usable


def _get_gpu_free_mb(host_id: str) -> int:
    result = _ssh_run(
        host_id,
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
        timeout=30,
    )
    return int(result.stdout.strip().splitlines()[0])


def _get_cpu_free_mb(host_id: str) -> int:
    result = _ssh_run(host_id, "free -m | awk '/^Mem:/{print $7}'", timeout=30)
    return int(result.stdout.strip().splitlines()[-1])


def _get_cpu_idle_pct(host_id: str) -> float:
    """Real CPU idle %, not load average. Load average was confirmed
    unreliable on server 95 during real S2DR4 runs (inflated by other
    processes' I/O wait; see s2dr4-run project's CLAUDE.md for the incident
    that established this) — `top`'s idle column reflects genuine spare CPU."""
    result = _ssh_run(host_id, "top -bn1 | grep '%Cpu(s)'", timeout=30)
    import re

    m = re.search(r"([\d.]+)\s*id", result.stdout or "")
    return float(m.group(1)) if m else 0.0


def _get_cpu_count(host_id: str) -> int:
    result = _ssh_run(host_id, "nproc", timeout=30)
    return int(result.stdout.strip().splitlines()[-1])


def _choose_gpu_concurrency(
    patch_km: float,
    free_gpu_mb: int,
    free_ram_mb: int,
    cpu_count: int,
) -> int:
    """Live, per-dispatch GPU concurrency -- NOT a fixed constant. Takes the
    minimum of what RAM, VRAM, and CPU cores each allow right now, so a
    quiet host can get 2-3 concurrent workers while a busy one correctly
    falls back to 1 (or 0), without anyone hand-tuning a number. Caller is
    responsible for checking cpu_idle_pct separately (same "genuinely busy"
    gate CPU mode already uses) before trusting this — a host mid-spike on
    someone else's job should never get MORE work added on top, no matter
    what this returns.

    All three constraints (RAM/VRAM/CPU) are measured, not guessed — see
    GPU_MEM_GB_PER_PATCH_KM / GPU_VRAM_GB_PER_WORKER docstrings above."""
    mem_per_worker_mb = GPU_MEM_GB_PER_PATCH_KM.get(
        patch_km, GPU_MEM_GB_PER_PATCH_KM[12.0]
    ) * 1024
    vram_per_worker_mb = GPU_VRAM_GB_PER_WORKER * 1024

    usable_ram_mb = free_ram_mb * GPU_SAFETY_FACTOR
    usable_vram_mb = free_gpu_mb * GPU_SAFETY_FACTOR
    by_ram = int(usable_ram_mb // mem_per_worker_mb)
    by_vram = int(usable_vram_mb // vram_per_worker_mb)

    if by_ram < 1 or by_vram < 1:
        return 0  # not even one worker fits right now

    by_cpu = max(1, cpu_count // GPU_MIN_THREADS_PER_WORKER)
    return max(1, min(by_ram, by_vram, by_cpu))


def _choose_cpu_patch_and_concurrency(free_mem_mb: int, cpu_count: int) -> tuple[float, int]:
    """Largest patch_km for which at least one worker fits in
    free_mem_mb * SAFETY_FACTOR, with concurrency = as many workers of that
    size as both memory AND cpu core budget allow (each worker gets >=1
    core; oversubscribing cores was the load-average-37 lockup incident in
    s2dr4-run's dev_tests/parallel_concurrency_v1)."""
    usable_mb = free_mem_mb * SAFETY_FACTOR
    for patch_km in sorted(CPU_MEM_GB_PER_PATCH_KM, reverse=True):
        mem_per_worker_mb = CPU_MEM_GB_PER_PATCH_KM[patch_km] * 1024
        by_mem = int(usable_mb // mem_per_worker_mb)
        if by_mem < 1:
            continue
        by_cpu = max(1, cpu_count - 1)  # leave >=1 core for the rest of the host
        concurrency = max(1, min(by_mem, by_cpu))
        return patch_km, concurrency
    return 0.0, 0  # not even the smallest patch fits right now


def probe_host(host_id: str, patch_km: float = GPU_PATCH_KM) -> HostCapacity:
    """Live capacity check for one host. Never raises — a host that's
    unreachable or too busy comes back with max_concurrency=0 so callers can
    just skip it, not crash the whole schedule evaluation.

    Call this right before EACH cell's dispatch, not once for the whole
    day's schedule tick — a number computed at 03:00 can be hours stale by
    the time a given cell actually reaches the front of the queue (see
    2026-08-14 discussion: a schedule-eval-time-only check can't react to
    a host getting busy mid-day)."""
    checked_at = time.time()
    if not host_configured(host_id):
        return HostCapacity(host_id, False, False, 0.0, 0, 0, checked_at,
                             error=f"SSH_HOST_{host_id} not set in .env")

    try:
        gpu_available = _check_gpu_usable(host_id)
        if gpu_available:
            free_gpu_mb = _get_gpu_free_mb(host_id)
            free_mem_mb = _get_cpu_free_mb(host_id)
            idle_pct = _get_cpu_idle_pct(host_id)
            cpu_count = _get_cpu_count(host_id)
            if idle_pct < 20.0:
                # Same "genuinely busy right now" gate as CPU mode — a host
                # some other process is currently loading shouldn't get MORE
                # work piled on just because GPU/RAM/VRAM look free on paper.
                return HostCapacity(host_id, True, True, patch_km, 0, free_mem_mb,
                                     checked_at, cpu_count=cpu_count,
                                     error=f"cpu_idle={idle_pct}% too busy")
            concurrency = _choose_gpu_concurrency(patch_km, free_gpu_mb, free_mem_mb, cpu_count)
            return HostCapacity(host_id, True, True, patch_km, concurrency,
                                 free_gpu_mb, checked_at, cpu_count=cpu_count)

        free_mem_mb = _get_cpu_free_mb(host_id)
        idle_pct = _get_cpu_idle_pct(host_id)
        cpu_count = _get_cpu_count(host_id)
        if idle_pct < 20.0:
            # Host is genuinely busy right now (real idle%, not load avg) —
            # don't add more work on top regardless of free memory.
            return HostCapacity(host_id, True, False, 0.0, 0, free_mem_mb, checked_at,
                                 cpu_count=cpu_count, error=f"cpu_idle={idle_pct}% too busy")
        patch_km, concurrency = _choose_cpu_patch_and_concurrency(free_mem_mb, cpu_count)
        return HostCapacity(host_id, True, False, patch_km, concurrency, free_mem_mb, checked_at,
                             cpu_count=cpu_count)

    except Exception as exc:  # noqa: BLE001 — an unreachable host must not crash the schedule
        return HostCapacity(host_id, False, False, 0.0, 0, 0, checked_at, error=str(exc))
