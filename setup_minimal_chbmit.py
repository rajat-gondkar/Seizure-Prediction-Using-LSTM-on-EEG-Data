#!/usr/bin/env python3
"""
Minimal CHB-MIT downloader & pipeline validator.

Downloads a tiny subset of the CHB-MIT dataset (patient chb01, 3 files ~80-150 MB total),
converts PhysioNet .seizures annotations into the .annotation.txt format expected by
the training scripts, creates CSV manifests, and validates that libDataIO can read them.
"""

import csv
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01"
DATA_DIR = Path("Data/CHB-MIT/chb01").resolve()
CSV_DIR  = Path("DataCSVs/CHB-MIT").resolve()

# A minimal subset that covers interictal + ictal for both train and test.
# chb01_01.edf  – interictal (no seizure)
# chb01_03.edf  – ictal (seizure at ~2996 s)
# chb01_04.edf  – ictal (seizure at ~1467 s)
TRAIN_FILES = ["chb01_01.edf", "chb01_03.edf"]
TEST_FILES  = ["chb01_04.edf"]
ALL_FILES   = list(dict.fromkeys(TRAIN_FILES + TEST_FILES))   # preserve order, unique

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def report_size(path: Path) -> str:
    mb = path.stat().st_size / (1024 * 1024)
    return f"{mb:.1f} MB"


def download(filename: str) -> bool:
    """Download a single file from PhysioNet CHB-MIT."""
    url = f"{BASE_URL}/{filename}?download"
    dest = DATA_DIR / filename

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ Already present: {filename} ({report_size(dest)})")
        return True

    print(f"  ↓ Downloading {filename} …")
    print(f"    URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("content-length", 0))
            if total:
                print(f"    Expected size: {total / (1024 * 1024):.1f} MB")
            with open(dest, "wb") as fh:
                fh.write(resp.read())
        print(f"    Saved → {dest} ({report_size(dest)})")
        return True
    except Exception as exc:
        print(f"    ✗ ERROR: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def convert_annotations(edf_name: str) -> None:
    """Read a .seizures annotation and write an .annotation.txt file."""
    edf_path   = DATA_DIR / edf_name
    anno_bin   = DATA_DIR / f"{edf_name}.seizures"
    anno_txt   = DATA_DIR / f"{edf_name}.annotation.txt"

    if not anno_bin.exists():
        print(f"  – No .seizures file for {edf_name} (will be treated as interictal)")
        return

    if anno_txt.exists():
        print(f"  ✓ Annotation text already present: {anno_txt.name}")
        return

    print(f"  ⟳ Converting annotations for {edf_name} …")

    # Sampling frequency from the EDF header
    try:
        import pyedflib
        with pyedflib.EdfReader(str(edf_path)) as fh:
            fs = float(fh.getSampleFrequencies()[0])
    except Exception as exc:
        print(f"    ✗ Could not read EDF header: {exc}")
        return

    # Read binary annotation via wfdb
    try:
        import wfdb
        ann = wfdb.rdann(str(edf_path), "seizures")
    except Exception as exc:
        print(f"    ✗ wfdb.rdann failed: {exc}")
        return

    if ann.ann_len == 0:
        print(f"    – Annotation file is empty")
        return

    if ann.ann_len % 2 != 0:
        print(f"    ⚠ Odd number of markers ({ann.ann_len}); pairing may be off")

    pairs = ann.ann_len // 2
    with open(anno_txt, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Onset", "Duration", "Annotation"])
        for i in range(0, ann.ann_len - 1, 2):
            start_samp = ann.sample[i]
            end_samp   = ann.sample[i + 1]
            onset      = start_samp / fs
            duration   = (end_samp - start_samp) / fs
            writer.writerow([f"{onset:.6f}", f"{duration:.6f}", "seizure"])

    print(f"    ✓ Created {anno_txt.name} ({pairs} seizure(s), fs={fs:.0f} Hz)")


def write_csv(file_list, csv_path: Path) -> None:
    """Write a CSV manifest with absolute paths."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename"])
        for name in file_list:
            writer.writerow([str(DATA_DIR / name)])
    print(f"  ✓ {csv_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("CHB-MIT Minimal Dataset Setup")
    print("=" * 65)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Download EDFs -------------------------------------------------
    print("\n[1/4] Downloading EEG recordings …")
    ok = all(download(f) for f in ALL_FILES)
    if not ok:
        print("\nSome downloads failed. Check your internet connection and")
        print("whether PhysioNet is reachable, then re-run this script.")
        sys.exit(1)

    # ---- 2. Download binary annotations -----------------------------------
    print("\n[2/4] Downloading binary annotation files …")
    for f in ALL_FILES:
        download(f + ".seizures")

    # ---- 3. Convert to text annotations -----------------------------------
    print("\n[3/4] Converting annotations to text format …")
    for f in ALL_FILES:
        convert_annotations(f)

    # ---- 4. Create CSV manifests ------------------------------------------
    print("\n[4/4] Creating CSV manifests …")
    write_csv(TRAIN_FILES, CSV_DIR / "chb01.csv")
    write_csv(TEST_FILES,  CSV_DIR / "chb01_Test.csv")

    # ---- 5. Validate against libDataIO ------------------------------------
    print("\n[5/4] Validating with libDataIO …")
    try:
        import libDataIO as dio
        train_files = dio.fnReadDataFileListCSV(str(CSV_DIR / "chb01.csv"))
        print(f"  ✓ Training CSV readable: {len(train_files)} file(s) listed")

        if train_files:
            (seg_label, data, duration, sfreq, channels, seq,
             n_ch, n_pts) = dio.fnReadEDFUsingPyEDFLib(
                 train_files[0], argPerformChecks=True, argNoData=True
             )
            print(f"  ✓ EDF header readable: {n_ch} ch, {sfreq} Hz, {duration:.0f} s")
    except Exception as exc:
        print(f"  ✗ Validation error: {exc}")
        print("    (This usually means a dependency is missing.)")

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("Done! Summary")
    print("=" * 65)
    print(f"\nData folder:  {DATA_DIR}")
    print(f"CSV folder:   {CSV_DIR}")
    print("\nFiles on disk:")
    for name in sorted(ALL_FILES):
        p = DATA_DIR / name
        print(f"  {name:<20} {report_size(p):>10}")
        anno = DATA_DIR / f"{name}.annotation.txt"
        if anno.exists():
            print(f"    └─ {anno.name}")
    print("\nCSV manifests:")
    print(f"  Training → {CSV_DIR / 'chb01.csv'}")
    print(f"  Testing  → {CSV_DIR / 'chb01_Test.csv'}")
    print("\nNext steps:")
    print("  1. Activate the venv:   source venv/bin/activate")
    print("  2. Install PyTorch:     pip install torch")
    print("  3. Run a smoke-test training (1 epoch, tiny batch):")
    print("     python scrTrainLSTM.py \\")
    print("       -csv ./DataCSVs/CHB-MIT/chb01.csv \\")
    print("       -tcsv ./DataCSVs/CHB-MIT/chb01_Test.csv \\")
    print("       -bs 2 -hd 64 -nl 1 -os 3 -lr 0.001 -ep 1 \\")
    print("       -smod 1 -smin -1 -smax 1")
    print()


if __name__ == "__main__":
    main()
