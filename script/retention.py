# S2_SuperResolution/utils/retention.py
#
# Daily housekeeping for the output-dir fix (core_runner._remote_out_dir is
# now scoped per (date, aoi_name) -- see that docstring): each cell's own
# scratch (TH/ raw patches etc.) now lives isolated forever unless something
# deletes it, so disk grows without bound. This is that something.
#
# Deliberately narrow: only ever touches a directory (at any depth up to
# max_depth) whose name is exactly a YYYY-MM-DD date, and only when that
# date is older than the retention window. Anything else -- the root-level
# `TH/` folder, loose root-level files like
# `S2SR10x_CORRIDOR-T47PNQ_..._TCI_1m.tif` (pre-existing production output
# from before this fix, not date-scoped at all), or a matched date dir's
# OWN children (PTTCORRIDOR-*, full/, priority/ -- never scanned for more
# dates) -- is left alone unconditionally; this script does not know what
# those are for and must not guess. Depth-bounded walk (not just direct
# children) because REMOTE_OUT_BASE nests as yyyy/mm/YYYY-MM-DD (see
# core_runner.remote_date_dir) while REMOTE_STAGE_ROOT is still flat
# (date dirs are direct children) -- one walk handles both shapes without
# the caller needing to know which root is which. /fs2 and /fs8 are both
# the same shared NFS mount everywhere (see core_runner.py), so this runs
# local-filesystem, no SSH needed.

import os
import re
import shutil
from datetime import date, timedelta

from core_runner import REMOTE_OUT_BASE, REMOTE_STAGE_ROOT

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _candidate_date_dirs(root: str, max_depth: int = 3) -> list[tuple[str, date]]:
    """Any directory up to `max_depth` levels under `root` whose name is
    exactly YYYY-MM-DD. Stops descending as soon as a date dir matches --
    its own children are never scanned for more dates."""
    out = []

    def _walk(d: str, depth: int) -> None:
        if depth > max_depth or not os.path.isdir(d):
            return
        for name in os.listdir(d):
            full = os.path.join(d, name)
            if not os.path.isdir(full):
                continue
            if _DATE_DIR_RE.match(name):
                try:
                    y, m, dd = (int(x) for x in name.split("-"))
                    out.append((full, date(y, m, dd)))
                except ValueError:
                    pass  # e.g. 2026-13-40 -- not a real date, leave it alone
                continue  # matched -- don't descend into it looking for more
            _walk(full, depth + 1)

    _walk(root, 1)
    return out


def plan_cleanup(retention_days: int = 14, today: date | None = None) -> dict:
    """Dry-run: returns what WOULD be deleted, deletes nothing. Always call
    this first and review the `to_delete` list before cleanup(dry_run=False)."""
    today = today or date.today()
    cutoff = today - timedelta(days=retention_days)

    result = {"retention_days": retention_days, "cutoff": cutoff.isoformat(), "roots": {}}
    for label, root in (("output", REMOTE_OUT_BASE), ("staging", REMOTE_STAGE_ROOT)):
        candidates = _candidate_date_dirs(root)
        to_delete = sorted(p for p, d in candidates if d < cutoff)
        to_keep = sorted(p for p, d in candidates if d >= cutoff)
        result["roots"][label] = {
            "root": root,
            "to_delete": to_delete,
            "to_keep": to_keep,
        }
    return result


def cleanup(retention_days: int = 14, dry_run: bool = True, today: date | None = None) -> dict:
    """Delete date-folders older than `retention_days`. dry_run=True (the
    default) never touches disk -- same output as plan_cleanup() plus a
    per-entry 'would_delete' log. Only dry_run=False actually calls
    shutil.rmtree, and only on paths that passed the exact YYYY-MM-DD
    same-regex check above."""
    plan = plan_cleanup(retention_days=retention_days, today=today)
    plan["dry_run"] = dry_run
    plan["deleted"] = []
    plan["errors"] = []

    if dry_run:
        return plan

    for label, info in plan["roots"].items():
        for path in info["to_delete"]:
            if not _DATE_DIR_RE.match(os.path.basename(path)):
                # belt-and-suspenders: refuse to ever rmtree anything whose
                # basename isn't a plain date, even if it somehow got into
                # this list.
                plan["errors"].append(f"refused (not a date dir): {path}")
                continue
            try:
                shutil.rmtree(path)
                plan["deleted"].append(path)
            except Exception as exc:  # noqa: BLE001
                plan["errors"].append(f"{path}: {type(exc).__name__}: {exc}")

    return plan
