#!/usr/bin/env python3
"""
Augment Entire_catalogue_chunk*.fits with 2MASS J/H/K photometry and errors
using Gaia DR3 tmass_psc_xsc_neighbourhood cross-match.

Writes new FITS files alongside the inputs, with a suffix (default ".1").
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack, join
from astroquery.gaia import Gaia

Gaia.ROW_LIMIT = -1

ADQL_QUERY = """
SELECT
    xm.source_id,
    xm.angular_distance AS tmass_ang_dist,
    xm.original_ext_source_id AS tmass_oid,
    tm.j_m      AS Jmag,
    tm.j_msigcom AS e_Jmag,
    tm.h_m      AS Hmag,
    tm.h_msigcom AS e_Hmag,
    tm.ks_m      AS Kmag,
    tm.ks_msigcom AS e_Kmag,
    tm.ph_qual  AS Qfl_2mass
FROM tap_upload.src AS s
JOIN gaiadr3.tmass_psc_xsc_neighbourhood AS xm
    ON s.source_id = xm.source_id
JOIN gaiadr1.tmass_original_valid AS tm
    ON tm.tmass_oid = xm.original_ext_source_id
"""

NEW_COLS = [
    ("Jmag", "f8"),
    ("e_Jmag", "f8"),
    ("Hmag", "f8"),
    ("e_Hmag", "f8"),
    ("Kmag", "f8"),
    ("e_Kmag", "f8"),
    ("Qfl_2mass", "U3"),
    ("tmass_ang_dist", "f8"),
    ("tmass_n_neigh", "i4"),
    ("tmass_oid", "i8"),
]


def _ensure_suffix(suffix: str) -> str:
    if not suffix:
        return ""
    return suffix if suffix.startswith(".") else f".{suffix}"


def _output_path(in_path: Path, suffix: str) -> Path:
    if in_path.suffix.lower() != ".fits":
        return in_path.with_name(in_path.name + suffix)
    return in_path.with_name(in_path.stem + suffix + in_path.suffix)


def _empty_best_table() -> Table:
    cols = {name: np.array([], dtype=dtype) for name, dtype in NEW_COLS}
    cols["source_id"] = np.array([], dtype="i8")
    return Table(cols)


def _best_from_neighbourhood(res: Table) -> Table:
    if len(res) == 0:
        return _empty_best_table()

    res.sort(["source_id", "tmass_ang_dist"])
    groups = res.group_by("source_id")
    start_idx = groups.groups.indices[:-1]
    counts = np.diff(groups.groups.indices)

    best = res[start_idx]
    best["tmass_n_neigh"] = counts.astype("i4")
    return best


def _fetch_best_for_batch(source_ids: np.ndarray, batch_idx: int, pause_s: float) -> Table:
    upload = Table([source_ids], names=["source_id"])
    job = Gaia.launch_job_async(
        ADQL_QUERY,
        upload_resource=upload,
        upload_table_name="src",
    )
    res = job.get_results()
    best = _best_from_neighbourhood(res)
    if pause_s > 0:
        time.sleep(pause_s)
    return best


def _augment_chunk(
    in_path: Path,
    out_path: Path,
    batch_size: int,
    pause_s: float,
    max_rows: int | None,
) -> None:
    print(f"\nReading: {in_path}")
    chunk = Table.read(in_path)
    if "source_id" not in chunk.colnames:
        raise RuntimeError(f"{in_path} is missing required column: source_id")

    if max_rows is not None:
        chunk = chunk[:max_rows]
        print(f"Limiting to first {max_rows} rows for testing.")

    source_ids = np.asarray(chunk["source_id"], dtype="i8")
    n_rows = len(source_ids)
    if n_rows == 0:
        print("No rows; writing empty output.")
        chunk.write(out_path, overwrite=True)
        return

    print(f"Fetching 2MASS matches in batches of {batch_size} (rows: {n_rows})")
    best_tables = []
    for i in range(0, n_rows, batch_size):
        batch = source_ids[i : i + batch_size]
        batch_idx = i // batch_size + 1
        print(f"  Batch {batch_idx}: {len(batch)} source_ids")
        best = _fetch_best_for_batch(batch, batch_idx, pause_s)
        if len(best) > 0:
            best_tables.append(best)

    if best_tables:
        best_all = vstack(best_tables)
    else:
        best_all = _empty_best_table()

    # Preserve original row order after join
    row_idx_col = "__row_idx__"
    while row_idx_col in chunk.colnames:
        row_idx_col = "_" + row_idx_col
    chunk[row_idx_col] = np.arange(len(chunk), dtype="i8")

    merged = join(chunk, best_all, keys="source_id", join_type="left")
    merged.sort(row_idx_col)
    merged.remove_column(row_idx_col)

    print(f"Writing: {out_path}")
    merged.write(out_path, overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add 2MASS J/H/K photometry to Entire_catalogue_chunk*.fits"
    )
    parser.add_argument(
        "--input-dir",
        default="BallTree_Xmatch",
        help="Directory containing the chunk FITS files",
    )
    parser.add_argument(
        "--pattern",
        default="Entire_catalogue_chunk*.fits",
        help="Glob pattern for input chunk FITS files",
    )
    parser.add_argument(
        "--suffix",
        default=".1",
        help="Suffix to append before .fits (default: .1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of source_ids per Gaia TAP upload batch",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=0.0,
        help="Seconds to sleep between batches (to be polite to the TAP service)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional: limit rows per chunk for testing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    suffix = _ensure_suffix(args.suffix)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {input_dir}")

    for in_path in files:
        out_path = _output_path(in_path, suffix)
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing: {out_path}")
            continue
        _augment_chunk(
            in_path=in_path,
            out_path=out_path,
            batch_size=args.batch_size,
            pause_s=args.pause_s,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()
