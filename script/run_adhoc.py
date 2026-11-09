"""Generic ad-hoc runner for a grid built with build_grid.py.

NOT used by production (`daily_run.py` has its own fixed grid/host/image/
output-path wiring and is untouched by this file). This is for anyone
cloning this repo to run their own AOI without risking a collision with
this team's production data or GPU host -- see the README's
"Before you run this yourself" table.

--image, --out-base, --stage-base and --host-id are REQUIRED (no defaults)
specifically so this can never silently fall back to this team's
production registry image, /fs8/.../Shared_data output path, or host 87.
They're applied by overriding core_runner's module-level constants at
runtime (not by editing core_runner.py) -- core_runner.py itself, which
Dagster's production job imports, is never touched by this script.

Usage:
    python run_adhoc.py GRID_PARQUET --dates 2026-01-10 2026-01-25 \\
        --host-id 95 --image myregistry/my-s2dr4:latest \\
        --out-base /my/output/path --stage-base /my/staging/path

Resumable (skips an aoi_name+date whose output TIF already exists) and
retries once on a transient failure before giving up on that cell.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time

import core_runner as r
import pandas as pd
from capacity_probe import probe_host

_TRANSIENT_MARKERS = (
    "IMCD_LZW", "imcd_lzw", "CORRUPT", "RasterioIOError", "Error opening",
    "HTTP response code", "timed out", "Connection", "CURL", "CPLE_",
)
_log_lock = threading.Lock()


def _is_transient(msg: str) -> bool:
    return bool(msg) and any(m in msg for m in _TRANSIENT_MARKERS)


def run_cell_once(host_id: str, aoi_name: str, aoi_file: str, date_str: str, patch_km: float):
    cap = probe_host(host_id, patch_km=patch_km)
    t0 = time.time()
    try:
        r.run_s2_super_resolution(
            host_id=host_id, aoi_file=aoi_file, date_str=date_str, aoi_name=aoi_name,
            patch_km=patch_km, gpu_available=cap.gpu_available, threads=cap.threads_per_worker,
        )
        r.validate_output(aoi_name=aoi_name, date_str=date_str)
        return True, round(time.time() - t0, 1), None
    except Exception as exc:  # noqa: BLE001
        return False, round(time.time() - t0, 1), f"{type(exc).__name__}: {exc}"


def worker(host_id, patch_km, log_path, work_q, counters, total_todo):
    while True:
        try:
            aoi_name, aoi_file, date_str = work_q.get_nowait()
        except queue.Empty:
            return
        ok, wall_s, err = False, 0.0, None
        for attempt in (1, 2):
            ok, wall_s, err = run_cell_once(host_id, aoi_name, aoi_file, date_str, patch_km)
            if ok or not _is_transient(err or "") or attempt == 2:
                break
            time.sleep(10)
        with _log_lock:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "date": date_str, "aoi_name": aoi_name, "wall_s": wall_s,
                    "ok": ok, "error_tail": (err or "")[-500:] if not ok else None,
                }) + "\n")
            counters["done"] += 1
            status = "OK" if ok else "FAIL"
            print(f"[{counters['done']}/{total_todo}] {date_str} {aoi_name} {status} ({wall_s:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid_parquet", help="grid.parquet from build_grid.py")
    ap.add_argument("--dates", nargs="+", required=True, help="YYYY-MM-DD, one or more")
    ap.add_argument("--host-id", required=True, help="<N> matching your SSH_HOST_<N> etc. in .env")
    ap.add_argument("--image", required=True, help="Your own docker image, e.g. myregistry/my-s2dr4:tag")
    ap.add_argument("--out-base", required=True, help="Your own output root (never this team's REMOTE_OUT_BASE)")
    ap.add_argument("--stage-base", required=True, help="Your own staging root (never this team's REMOTE_STAGE_ROOT)")
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="Cap on workers; default lets capacity_probe decide per host")
    ap.add_argument("--log", default="run_adhoc_log.jsonl", help="Per-cell result log (default: run_adhoc_log.jsonl)")
    args = ap.parse_args()

    # Override core_runner's module constants for THIS process only --
    # core_runner.py on disk is never modified.
    r.IMAGE = args.image
    r.REMOTE_OUT_BASE = args.out_base
    r.REMOTE_STAGE_ROOT = args.stage_base

    df = pd.read_parquet(args.grid_parquet)
    work_q: "queue.Queue" = queue.Queue()
    n_skip = 0
    for date_str in args.dates:
        for _, row in df.iterrows():
            if r.output_already_done(row["aoi_name"], date_str):
                n_skip += 1
                continue
            work_q.put((row["aoi_name"], row["aoi_file"], date_str))
    total_all = len(df) * len(args.dates)
    total_todo = total_all - n_skip
    print(f"total {total_all}, already done {n_skip}, remaining {total_todo}", flush=True)
    print(f"writing output under {args.out_base} (not this team's production path)", flush=True)

    patch_km = float(df["patch_km"].iloc[0]) if len(df) else 8.0
    counters = {"done": 0}
    active: list[threading.Thread] = []
    last_target = None
    while not work_q.empty() or any(t.is_alive() for t in active):
        active = [t for t in active if t.is_alive()]
        if work_q.empty():
            for t in active:
                t.join()
            break
        cap = probe_host(args.host_id, patch_km=patch_km)
        target = cap.max_concurrency if cap.reachable else 0
        if args.max_concurrency is not None:
            target = min(target, args.max_concurrency)
        if target != last_target:
            print(f"[capacity] max_concurrency={target} (active={len(active)})", flush=True)
            last_target = target
        while len(active) < target and not work_q.empty():
            t = threading.Thread(target=worker, args=(args.host_id, patch_km, args.log, work_q, counters, total_todo), daemon=True)
            t.start()
            active.append(t)
        time.sleep(30)

    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
