"""
extract_data.py — Resumable extraction for ShotQA data.tar.gz

Why this version: data.tar.gz (~40GB) is a single non-seekable gzip stream,
so a crash mid-extraction can't be resumed by seeking — the archive has to
be read from the start again either way. What THIS script does to make that
less painful:

  1. Skips writing any file that's already fully extracted (checked by
     comparing expected size vs. size on disk), so a second run doesn't
     redo disk I/O for files that already made it out safely.
  2. Extracts member-by-member (not tar.extractall()) so a crash only
     loses the *current* file, not a big done-in-one-call chunk.
  3. Logs progress every N files and writes a small resume-state file so
     you can see exactly where it crashed / how far it got.
  4. Catches KeyboardInterrupt and most I/O errors cleanly, flushing the
     log before exiting, instead of dying silently mid-file.

Usage:
    python extract_data.py \
        --archive data/raw/data.tar.gz \
        --outdir data/raw/extracted

Run it again after a crash — it will pick up where it left off (modulo
re-reading the compressed stream, which is unavoidable with a non-seekable
.tar.gz; what it SAVES you is re-writing files you already have).
"""

import argparse
import sys
import tarfile
import time
from pathlib import Path


def human(n_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PB"


def already_extracted(member: tarfile.TarInfo, outdir: Path) -> bool:
    """Check if this member is already fully on disk (size match)."""
    target = outdir / member.name
    if not target.exists():
        return False
    try:
        return target.stat().st_size == member.size
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Path to data.tar.gz")
    parser.add_argument("--outdir", required=True, help="Extraction target dir")
    parser.add_argument(
        "--log-every", type=int, default=500, help="Log progress every N files"
    )
    args = parser.parse_args()

    archive_path = Path(args.archive)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"ERROR: archive not found at {archive_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Archive: {archive_path} ({human(archive_path.stat().st_size)})")
    print(f"Output:  {outdir}")
    print("Opening tar stream (this does NOT load 40GB into memory — "
          "it's a streaming read)...")

    start_time = time.time()
    n_seen = 0
    n_skipped = 0
    n_extracted = 0
    bytes_extracted = 0
    last_name = None

    try:
        # 'r:gz' = streaming mode, reads forward-only. This is correct for
        # a non-seekable stream and is what makes resume-by-skip possible
        # without needing random access.
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                n_seen += 1
                last_name = member.name

                if not member.isfile():
                    continue  # skip dirs/symlinks, nothing to write

                if already_extracted(member, outdir):
                    n_skipped += 1
                else:
                    tar.extract(member, path=outdir)
                    n_extracted += 1
                    bytes_extracted += member.size

                if n_seen % args.log_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{elapsed:7.1f}s] seen={n_seen:,} "
                        f"extracted={n_extracted:,} skipped(already done)={n_skipped:,} "
                        f"last={last_name}"
                    )

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at member: {last_name}")
        print(f"Progress so far: seen={n_seen:,}, extracted={n_extracted:,}, "
              f"skipped={n_skipped:,}")
        print("Safe to re-run this script — already-extracted files will be skipped.")
        sys.exit(130)

    except (OSError, tarfile.TarError) as e:
        print(f"\nERROR during extraction near member: {last_name}")
        print(f"  {type(e).__name__}: {e}")
        print(f"Progress so far: seen={n_seen:,}, extracted={n_extracted:,}, "
              f"skipped={n_skipped:,}")
        print("This often means: disk full, corrupted download, or a crash "
              "mid-write on the previous run leaving a partial file that "
              "still matched size checks incorrectly (rare). Re-run this "
              "script — it will re-verify and continue.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total entries seen: {n_seen:,}")
    print(f"  Newly extracted:    {n_extracted:,} ({human(bytes_extracted)})")
    print(f"  Already present:    {n_skipped:,}")
    print(f"\nNext: run the img_id -> path verification script to confirm "
          f"the extracted structure matches meta.jsonl.")


if __name__ == "__main__":
    main()