#!/usr/bin/env python3
"""
validate_dataset.py
===================
Pre-flight check for your CHB-MIT dataset.

Scans every EDF file listed in a CSV manifest, attempts to open it with
pyedflib, and reports any corrupted / truncated / unreadable files.

Usage
-----
Validate a single patient CSV:
    python validate_dataset.py -csv ./DataCSVs/CHB-MIT/chb01.csv

Validate all downloaded data:
    python validate_dataset.py -csv ./DataCSVs/CHB-MIT/all_patients_train.csv

A corrupted file will show:
    ✗ chb01_03.edf  — OSError: cannot read header

Clean files show:
    ✓ chb01_01.edf  — 23 ch, 256 Hz, 3600 s, 40.4 MB
"""

import argparse
import csv
import os
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def validate_file(fpath: str) -> tuple:
    """Try to open an EDF and return (ok, info_string or error_string)."""
    try:
        import pyedflib
        with pyedflib.EdfReader(fpath) as fh:
            n_ch = fh.signals_in_file
            sf = fh.getSampleFrequencies()[0]
            dur = fh.file_duration
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            info = f"{n_ch} ch, {sf:.0f} Hz, {dur} s, {size_mb:.1f} MB"
        return True, info
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Validate CHB-MIT EDF files")
    parser.add_argument("-csv", "--csvpath", required=True, help="Path to CSV manifest")
    args = parser.parse_args()

    # Read CSV
    files = []
    with open(args.csvpath, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if row:
                files.append(row[0])

    print(f"Validating {len(files)} EDF files from {args.csvpath}\n")

    ok_count = 0
    bad_count = 0
    bad_files = []

    iter_files = tqdm(files, desc="Validating", unit="file", ncols=80) if tqdm else files

    for fpath in iter_files:
        fname = os.path.basename(fpath)
        ok, msg = validate_file(fpath)
        if ok:
            ok_count += 1
            print(f"  ✓ {fname:<25s} — {msg}")
        else:
            bad_count += 1
            bad_files.append((fname, msg))
            print(f"  ✗ {fname:<25s} — {msg}")

    print(f"\n{'='*60}")
    print(f"Results: {ok_count} OK, {bad_count} BAD out of {len(files)} files")

    if bad_files:
        print("\nCorrupted / unreadable files:")
        for fname, msg in bad_files:
            print(f"  ✗ {fname} — {msg}")
        print("\nRecommendation: delete the bad files and re-run setup_chbmit.py")
        print("  (the script will re-download only missing files)")
    else:
        print("\nAll files look good! You can start training safely.")


if __name__ == "__main__":
    main()
