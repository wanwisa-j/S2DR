"""Generate a lightweight preview PNG of the whole corridor from a full/priority
VRT, for sharing on a platform (not for analysis -- see docstring below).

The corridor is long, thin, and diagonal across its bounding box (per-cell
valid coverage was ~7-40% in validation), so a plain flat-background render
would show mostly empty space as solid black. Transparency (alpha from
nodata) keeps it readable when the platform composites it over a basemap.

Usage:
    python3 generate_preview.py <input.vrt> <output.png> [--max-px 3000]
"""
import argparse
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Matches gdalwarp's error line for a tile it couldn't decode, e.g.:
#   ERROR 1: /path/PTTCORRIDOR-00128/S2SR10x_..._TCI_1m.tif, band 1: IReadBlock failed...
# gdalwarp reports this by literally joining the VRT's own dir with the
# source's on-disk (possibly ../-relative) path, e.g. ".../full/../PTTCORRIDOR-00128/...";
# normpath collapses that to the same absolute form _rebuild_excluding produces.
_BAD_TILE_RE = re.compile(r"ERROR \d+: (\S+\.tif), band \d+:")


def _rebuild_excluding(vrt_path: str, exclude: set[str], tmp: str) -> str:
    """A corrupt source tile fails the WHOLE gdalwarp read, not just its own
    cell (seen for real: one bad tile among hundreds silently kills the
    entire month's preview otherwise). Drop the known-bad file(s) and
    rebuild a scratch VRT from what's left, so one dead cell doesn't cost
    the whole corridor's preview -- the platform gets everything else.

    SourceFilename entries in the VRT are relative to the VRT's OWN
    directory (e.g. "../PTTCORRIDOR-00047/....tif"), not to this process's
    cwd -- must resolve to absolute before handing them to gdalbuildvrt, or
    it fails to find every single one of them."""
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    tree = ET.parse(vrt_path)
    tifs = sorted({
        os.path.normpath(os.path.join(vrt_dir, el.text))
        for el in tree.getroot().iter("SourceFilename")
    } - exclude)
    rebuilt = f"{tmp}/rebuilt.vrt"
    subprocess.run(["gdalbuildvrt", rebuilt, *tifs], check=True, capture_output=True)
    return rebuilt


def generate_preview(vrt_path: str, png_path: str, max_px: int = 3000) -> None:
    import rasterio

    with rasterio.open(vrt_path) as ds:
        w, h = ds.width, ds.height
    if w >= h:
        out_w, out_h = max_px, round(max_px * h / w)
    else:
        out_w, out_h = round(max_px * w / h), max_px

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        source = vrt_path
        with_alpha = f"{tmp}/with_alpha.tif"
        # Up to 5 corrupt tiles excluded before giving up -- one bad file is
        # the observed case (4/7 days in Sept 2026, always exactly one), more
        # than that suggests something worse than isolated corruption.
        for _ in range(5):
            # One step, via gdalwarp not gdal_translate: gdal_translate's -r
            # average blends nodata=0 straight into the mean with real data
            # (verified: 2026-09-15 test came out 100% opaque, corridor
            # pixels diluted to near-black) -- at ~27x decimation each output
            # pixel averages ~750 source pixels, nearly all nodata on a
            # corridor this sparse, so real signal all but vanishes.
            # gdalwarp's -srcnodata excludes nodata from the average instead
            # (real corridor pixels average only with each other, full
            # brightness preserved) and -dstalpha then encodes HOW MUCH of
            # each output cell was real coverage as a soft alpha, rather
            # than a hard cutoff.
            result = subprocess.run([
                "gdalwarp", "-ts", str(out_w), str(out_h),
                "-r", "average", "-srcnodata", "0", "-dstalpha",
                "-overwrite", source, with_alpha,
            ], capture_output=True)
            if result.returncode == 0:
                break
            stderr = result.stderr.decode(errors="replace")
            bad = _BAD_TILE_RE.search(stderr)
            if not bad:
                raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
            skipped.append(os.path.normpath(bad.group(1)))
            source = _rebuild_excluding(vrt_path, set(skipped), tmp)
        else:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

        subprocess.run(
            ["gdal_translate", "-of", "PNG", with_alpha, png_path],
            check=True, capture_output=True,
        )
    return skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vrt_path")
    p.add_argument("png_path")
    p.add_argument("--max-px", type=int, default=3000, help="long-edge size in pixels (default 3000)")
    args = p.parse_args()
    skipped = generate_preview(args.vrt_path, args.png_path, args.max_px)
    size_kb = Path(args.png_path).stat().st_size / 1024
    print(f"wrote {args.png_path} ({size_kb:.0f} KB)")
    if skipped:
        print(f"skipped {len(skipped)} corrupt source tile(s): {skipped}")


if __name__ == "__main__":
    main()
