#!/usr/bin/env python3
"""
CHB-MIT Dataset Setup — Minimal or Full Download
=================================================
Downloads the CHB-MIT scalp EEG dataset from PhysioNet and prepares it for
training / testing with the LSTM pipeline.

Modes
-----
--mode minimal   : ~120 MB, 3 files (patient chb01) — quick smoke test
--mode full      : ~40–45 GB, all 22 patients — real training

What it does
------------
1. Downloads EDF recordings (and .seizures annotation binaries)
2. Converts .seizures → .annotation.txt for each file
3. Creates per-patient CSV manifests:
     DataCSVs/CHB-MIT/chbXX.csv       → first 80 % of files (train)
     DataCSVs/CHB-MIT/chbXX_test.csv  → last 20 % of files (test)
4. Optionally creates combined multi-patient CSVs

Usage
-----
Interactive (asks what you want):
    python3 setup_chbmit.py

Non-interactive — minimal subset:
    python3 setup_chbmit.py --mode minimal

Non-interactive — full dataset (all patients):
    python3 setup_chbmit.py --mode full --yes

Non-interactive — specific patients only:
    python3 setup_chbmit.py --mode full --patients chb01,chb02,chb03 --yes
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Graceful fallback if tqdm not installed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
DATA_DIR_DEFAULT = "Data/CHB-MIT"
CSV_DIR_DEFAULT = "DataCSVs/CHB-MIT"

# Minimal subset (smoke test)
MINIMAL_FILES = {
    "chb01": ["chb01_01.edf", "chb01_03.edf", "chb01_04.edf"]
}

# Full dataset patients (chb01–chb24, skipping chb21 which doesn't exist)
ALL_PATIENTS = [f"chb{i:02d}" for i in range(1, 25) if f"chb{i:02d}" != "chb21"]

# Approx. total size for the full dataset (GB) — used for the warning
FULL_SIZE_GB = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report_size(path: Path) -> str:
    mb = path.stat().st_size / (1024 * 1024)
    return f"{mb:.1f} MB"


def human_size(bytes_val: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def download(url: str, dest: Path, timeout: int = 300, show_progress: bool = True) -> bool:
    """Download a single file. Validates size against Content-Length."""
    # If file exists, check size against expected size from a HEAD request
    try:
        req_head = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        with urllib.request.urlopen(req_head, timeout=timeout) as resp_head:
            expected_size = int(resp_head.headers.get("content-length", 0))
    except Exception:
        expected_size = 0

    if dest.exists() and dest.stat().st_size > 0:
        if expected_size > 0 and dest.stat().st_size == expected_size:
            return True  # Already present and correct size
        elif expected_size == 0:
            return True  # Can't verify, assume OK
        else:
            print(f"  ⚠ {dest.name} exists but wrong size ({dest.stat().st_size} != {expected_size}), re-downloading...")
            dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh:
                if tqdm and show_progress and total > 0:
                    pbar = tqdm(total=total, unit="B", unit_scale=True,
                                unit_divisor=1024, desc=dest.name, leave=False)
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        fh.write(chunk)
                        pbar.update(len(chunk))
                    pbar.close()
                else:
                    fh.write(resp.read())

        # Verify downloaded size
        if total > 0 and dest.stat().st_size != total:
            print(f"    ✗ SIZE MISMATCH: {dest.name} ({dest.stat().st_size} != {total})")
            dest.unlink()
            return False

        return True
    except Exception as exc:
        print(f"    ✗ ERROR downloading {dest.name}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def fetch_records_list() -> list:
    """
    Try to fetch the PhysioNet RECORDS file to discover all EDF filenames.
    Returns a list like ['chb01/chb01_01.edf', ...].
    Falls back to a hardcoded minimal list if fetching fails.
    """
    records_url = f"{BASE_URL}/RECORDS"
    records_file = Path(".chbmit_records")

    # Try cache first
    if records_file.exists():
        with open(records_file) as fh:
            lines = [ln.strip() for ln in fh if ln.strip().endswith(".edf")]
        if lines:
            return lines

    print("Fetching file list from PhysioNet RECORDS …")
    try:
        req = urllib.request.Request(records_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().endswith(".edf")]
        with open(records_file, "w") as fh:
            fh.write(text)
        return lines
    except Exception as exc:
        print(f"  ⚠ Could not fetch RECORDS ({exc}).")
        print("  Falling back to hardcoded file list.")
        # Hardcoded fallback — covers the complete known dataset
        return _fallback_records()


def _fallback_records() -> list:
    """Hardcoded CHB-MIT record list (generated from the published dataset)."""
    records = []
    # chb01
    for i in range(1, 47):
        records.append(f"chb01/chb01_{i:02d}.edf")
    # chb02
    for i in range(1, 17):
        records.append(f"chb02/chb02_{i:02d}.edf")
    records.append("chb02/chb02_16+.edf")
    # chb03
    for i in range(1, 34):
        records.append(f"chb03/chb03_{i:02d}.edf")
    # chb04
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47):
        records.append(f"chb04/chb04_{i:02d}.edf")
    # chb05
    for i in range(1, 40):
        records.append(f"chb05/chb05_{i:02d}.edf")
    # chb06
    for i in range(1, 25):
        records.append(f"chb06/chb06_{i:02d}.edf")
    records.append("chb06/chb06_24+.edf")
    records.append("chb06/chb06_25+.edf")
    # chb07
    for i in range(1, 20):
        records.append(f"chb07/chb07_{i:02d}.edf")
    # chb08
    for i in range(1, 30):
        records.append(f"chb08/chb08_{i:02d}.edf")
    records.append("chb08/chb08_29+.edf")
    # chb09
    for i in range(1, 20):
        records.append(f"chb09/chb09_{i:02d}.edf")
    # chb10
    for i in range(1, 51):
        records.append(f"chb10/chb10_{i:02d}.edf")
    records.append("chb10/chb10_50+.edf")
    # chb11
    for i in range(1, 36):
        records.append(f"chb11/chb11_{i:02d}.edf")
    # chb12
    for i in range(1, 28):
        records.append(f"chb12/chb12_{i:02d}.edf")
    for i in range(27, 39):
        records.append(f"chb12/chb12_{i:02d}.edf")
    # chb13
    for i in range(1, 44):
        records.append(f"chb13/chb13_{i:02d}.edf")
    records.append("chb13/chb13_43+.edf")
    # chb14
    for i in range(1, 30):
        records.append(f"chb14/chb14_{i:02d}.edf")
    # chb15
    for i in range(1, 37):
        records.append(f"chb15/chb15_{i:02d}.edf")
    # chb16 (often called chb16, actually present as chb16_01 etc.)
    for i in range(1, 20):
        records.append(f"chb16/chb16_{i:02d}.edf")
    # chb17
    for i in range(1, 9):
        records.append(f"chb17/chb17_{i:02d}.edf")
    for i in range(31, 37):
        records.append(f"chb17/chb17_{i:02d}.edf")
    for i in range(39, 44):
        records.append(f"chb17/chb17_{i:02d}.edf")
    # chb18
    for i in range(1, 37):
        records.append(f"chb18/chb18_{i:02d}.edf")
    # chb19
    for i in range(1, 31):
        records.append(f"chb19/chb19_{i:02d}.edf")
    # chb20
    for i in range(1, 14):
        records.append(f"chb20/chb20_{i:02d}.edf")
    records.append("chb20/chb20_12+.edf")
    records.append("chb20/chb20_13+.edf")
    records.append("chb20/chb20_14+.edf")
    records.append("chb20/chb20_15+.edf")
    records.append("chb20/chb20_16+.edf")
    records.append("chb20/chb20_17+.edf")
    records.append("chb20/chb20_18+.edf")
    records.append("chb20/chb20_19+.edf")
    records.append("chb20/chb20_20+.edf")
    records.append("chb20/chb20_21+.edf")
    records.append("chb20/chb20_22+.edf")
    records.append("chb20/chb20_23+.edf")
    records.append("chb20/chb20_24+.edf")
    records.append("chb20/chb20_25+.edf")
    records.append("chb20/chb20_26+.edf")
    records.append("chb20/chb20_27+.edf")
    records.append("chb20/chb20_28+.edf")
    records.append("chb20/chb20_29+.edf")
    records.append("chb20/chb20_30+.edf")
    records.append("chb20/chb20_31+.edf")
    records.append("chb20/chb20_32+.edf")
    records.append("chb20/chb20_33+.edf")
    records.append("chb20/chb20_34+.edf")
    records.append("chb20/chb20_35+.edf")
    records.append("chb20/chb20_36+.edf")
    records.append("chb20/chb20_37+.edf")
    records.append("chb20/chb20_38+.edf")
    records.append("chb20/chb20_39+.edf")
    records.append("chb20/chb20_40+.edf")
    records.append("chb20/chb20_41+.edf")
    records.append("chb20/chb20_42+.edf")
    records.append("chb20/chb20_43+.edf")
    records.append("chb20/chb20_44+.edf")
    records.append("chb20/chb20_45+.edf")
    records.append("chb20/chb20_46+.edf")
    records.append("chb20/chb20_47+.edf")
    records.append("chb20/chb20_48+.edf")
    records.append("chb20/chb20_49+.edf")
    records.append("chb20/chb20_50+.edf")
    records.append("chb20/chb20_51+.edf")
    records.append("chb20/chb20_52+.edf")
    records.append("chb20/chb20_53+.edf")
    records.append("chb20/chb20_54+.edf")
    records.append("chb20/chb20_55+.edf")
    records.append("chb20/chb20_56+.edf")
    records.append("chb20/chb20_57+.edf")
    records.append("chb20/chb20_58+.edf")
    records.append("chb20/chb20_59+.edf")
    records.append("chb20/chb20_60+.edf")
    records.append("chb20/chb20_61+.edf")
    records.append("chb20/chb20_62+.edf")
    records.append("chb20/chb20_63+.edf")
    records.append("chb20/chb20_64+.edf")
    records.append("chb20/chb20_65+.edf")
    records.append("chb20/chb20_66+.edf")
    records.append("chb20/chb20_67+.edf")
    # chb22
    for i in range(1, 11):
        records.append(f"chb22/chb22_{i:02d}.edf")
    records.append("chb22/chb22_10+.edf")
    records.append("chb22/chb22_11+.edf")
    records.append("chb22/chb22_12+.edf")
    records.append("chb22/chb22_13+.edf")
    records.append("chb22/chb22_14+.edf")
    records.append("chb22/chb22_15+.edf")
    records.append("chb22/chb22_16+.edf")
    records.append("chb22/chb22_17+.edf")
    records.append("chb22/chb22_18+.edf")
    records.append("chb22/chb22_19+.edf")
    records.append("chb22/chb22_20+.edf")
    records.append("chb22/chb22_21+.edf")
    records.append("chb22/chb22_22+.edf")
    records.append("chb22/chb22_23+.edf")
    records.append("chb22/chb22_24+.edf")
    records.append("chb22/chb22_25+.edf")
    records.append("chb22/chb22_26+.edf")
    records.append("chb22/chb22_27+.edf")
    records.append("chb22/chb22_28+.edf")
    records.append("chb22/chb22_29+.edf")
    records.append("chb22/chb22_30+.edf")
    records.append("chb22/chb22_31+.edf")
    records.append("chb22/chb22_32+.edf")
    records.append("chb22/chb22_33+.edf")
    records.append("chb22/chb22_34+.edf")
    records.append("chb22/chb22_35+.edf")
    records.append("chb22/chb22_36+.edf")
    records.append("chb22/chb22_37+.edf")
    records.append("chb22/chb22_38+.edf")
    # chb23
    for i in range(1, 28):
        records.append(f"chb23/chb23_{i:02d}.edf")
    records.append("chb23/chb23_06+.edf")
    records.append("chb23/chb23_07+.edf")
    records.append("chb23/chb23_08+.edf")
    records.append("chb23/chb23_09+.edf")
    # chb24
    for i in range(1, 12):
        records.append(f"chb24/chb24_{i:02d}.edf")
    records.append("chb24/chb24_11+.edf")
    records.append("chb24/chb24_12+.edf")
    records.append("chb24/chb24_13+.edf")
    records.append("chb24/chb24_14+.edf")
    records.append("chb24/chb24_15+.edf")
    records.append("chb24/chb24_16+.edf")
    records.append("chb24/chb24_17+.edf")
    records.append("chb24/chb24_18+.edf")
    records.append("chb24/chb24_19+.edf")
    records.append("chb24/chb24_20+.edf")
    records.append("chb24/chb24_21+.edf")
    records.append("chb24/chb24_22+.edf")
    records.append("chb24/chb24_23+.edf")
    records.append("chb24/chb24_24+.edf")
    records.append("chb24/chb24_25+.edf")
    records.append("chb24/chb24_26+.edf")
    records.append("chb24/chb24_27+.edf")
    records.append("chb24/chb24_28+.edf")
    records.append("chb24/chb24_29+.edf")
    records.append("chb24/chb24_30+.edf")
    records.append("chb24/chb24_31+.edf")
    records.append("chb24/chb24_32+.edf")
    records.append("chb24/chb24_33+.edf")
    records.append("chb24/chb24_34+.edf")
    records.append("chb24/chb24_35+.edf")
    records.append("chb24/chb24_36+.edf")
    records.append("chb24/chb24_37+.edf")
    records.append("chb24/chb24_38+.edf")
    records.append("chb24/chb24_39+.edf")
    records.append("chb24/chb24_40+.edf")
    records.append("chb24/chb24_41+.edf")
    records.append("chb24/chb24_42+.edf")
    records.append("chb24/chb24_43+.edf")
    records.append("chb24/chb24_44+.edf")
    records.append("chb24/chb24_45+.edf")
    records.append("chb24/chb24_46+.edf")
    records.append("chb24/chb24_47+.edf")
    records.append("chb24/chb24_48+.edf")
    records.append("chb24/chb24_49+.edf")
    records.append("chb24/chb24_50+.edf")
    records.append("chb24/chb24_51+.edf")
    records.append("chb24/chb24_52+.edf")
    records.append("chb24/chb24_53+.edf")
    records.append("chb24/chb24_54+.edf")
    records.append("chb24/chb24_55+.edf")
    records.append("chb24/chb24_56+.edf")
    records.append("chb24/chb24_57+.edf")
    records.append("chb24/chb24_58+.edf")
    records.append("chb24/chb24_59+.edf")
    records.append("chb24/chb24_60+.edf")
    records.append("chb24/chb24_61+.edf")
    records.append("chb24/chb24_62+.edf")
    records.append("chb24/chb24_63+.edf")
    records.append("chb24/chb24_64+.edf")
    records.append("chb24/chb24_65+.edf")
    records.append("chb24/chb24_66+.edf")
    records.append("chb24/chb24_67+.edf")
    records.append("chb24/chb24_68+.edf")
    records.append("chb24/chb24_69+.edf")
    records.append("chb24/chb24_70+.edf")
    records.append("chb24/chb24_71+.edf")
    records.append("chb24/chb24_72+.edf")
    records.append("chb24/chb24_73+.edf")
    records.append("chb24/chb24_74+.edf")
    records.append("chb24/chb24_75+.edf")
    records.append("chb24/chb24_76+.edf")
    records.append("chb24/chb24_77+.edf")
    records.append("chb24/chb24_78+.edf")
    records.append("chb24/chb24_79+.edf")
    records.append("chb24/chb24_80+.edf")
    records.append("chb24/chb24_81+.edf")
    records.append("chb24/chb24_82+.edf")
    records.append("chb24/chb24_83+.edf")
    records.append("chb24/chb24_84+.edf")
    records.append("chb24/chb24_85+.edf")
    records.append("chb24/chb24_86+.edf")
    records.append("chb24/chb24_87+.edf")
    records.append("chb24/chb24_88+.edf")
    records.append("chb24/chb24_89+.edf")
    records.append("chb24/chb24_90+.edf")
    records.append("chb24/chb24_91+.edf")
    records.append("chb24/chb24_92+.edf")
    records.append("chb24/chb24_93+.edf")
    records.append("chb24/chb24_94+.edf")
    records.append("chb24/chb24_95+.edf")
    records.append("chb24/chb24_96+.edf")
    records.append("chb24/chb24_97+.edf")
    records.append("chb24/chb24_98+.edf")
    records.append("chb24/chb24_99+.edf")
    records.append("chb24/chb24_100+.edf")
    records.append("chb24/chb24_101+.edf")
    records.append("chb24/chb24_102+.edf")
    records.append("chb24/chb24_103+.edf")
    records.append("chb24/chb24_104+.edf")
    records.append("chb24/chb24_105+.edf")
    records.append("chb24/chb24_106+.edf")
    records.append("chb24/chb24_107+.edf")
    records.append("chb24/chb24_108+.edf")
    records.append("chb24/chb24_109+.edf")
    records.append("chb24/chb24_110+.edf")
    records.append("chb24/chb24_111+.edf")
    records.append("chb24/chb24_112+.edf")
    records.append("chb24/chb24_113+.edf")
    records.append("chb24/chb24_114+.edf")
    records.append("chb24/chb24_115+.edf")
    records.append("chb24/chb24_116+.edf")
    records.append("chb24/chb24_117+.edf")
    records.append("chb24/chb24_118+.edf")
    records.append("chb24/chb24_119+.edf")
    records.append("chb24/chb24_120+.edf")
    records.append("chb24/chb24_121+.edf")
    records.append("chb24/chb24_122+.edf")
    records.append("chb24/chb24_123+.edf")
    records.append("chb24/chb24_124+.edf")
    records.append("chb24/chb24_125+.edf")
    records.append("chb24/chb24_126+.edf")
    records.append("chb24/chb24_127+.edf")
    records.append("chb24/chb24_128+.edf")
    records.append("chb24/chb24_129+.edf")
    records.append("chb24/chb24_130+.edf")
    records.append("chb24/chb24_131+.edf")
    records.append("chb24/chb24_132+.edf")
    records.append("chb24/chb24_133+.edf")
    records.append("chb24/chb24_134+.edf")
    records.append("chb24/chb24_135+.edf")
    records.append("chb24/chb24_136+.edf")
    records.append("chb24/chb24_137+.edf")
    records.append("chb24/chb24_138+.edf")
    records.append("chb24/chb24_139+.edf")
    records.append("chb24/chb24_140+.edf")
    records.append("chb24/chb24_141+.edf")
    records.append("chb24/chb24_142+.edf")
    records.append("chb24/chb24_143+.edf")
    records.append("chb24/chb24_144+.edf")
    records.append("chb24/chb24_145+.edf")
    records.append("chb24/chb24_146+.edf")
    records.append("chb24/chb24_147+.edf")
    records.append("chb24/chb24_148+.edf")
    records.append("chb24/chb24_149+.edf")
    records.append("chb24/chb24_150+.edf")
    return records


def convert_annotations(edf_path: Path) -> bool:
    """Read a .seizures annotation and write an .annotation.txt file."""
    anno_bin = edf_path.with_suffix(edf_path.suffix + ".seizures")
    anno_txt = edf_path.with_suffix(edf_path.suffix + ".annotation.txt")

    if not anno_bin.exists():
        return False  # No seizures → no annotation needed

    if anno_txt.exists():
        return True  # Already converted

    print(f"  ⟳ Converting annotations for {edf_path.name} …")

    try:
        import pyedflib
        with pyedflib.EdfReader(str(edf_path)) as fh:
            fs = float(fh.getSampleFrequencies()[0])
    except Exception as exc:
        print(f"    ✗ Could not read EDF header: {exc}")
        return False

    try:
        import wfdb
        ann = wfdb.rdann(str(edf_path), "seizures")
    except Exception as exc:
        print(f"    ✗ wfdb.rdann failed: {exc}")
        return False

    if ann.ann_len == 0:
        return False

    pairs = ann.ann_len // 2
    with open(anno_txt, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Onset", "Duration", "Annotation"])
        for i in range(0, ann.ann_len - 1, 2):
            start_samp = ann.sample[i]
            end_samp = ann.sample[i + 1]
            onset = start_samp / fs
            duration = (end_samp - start_samp) / fs
            writer.writerow([f"{onset:.6f}", f"{duration:.6f}", "seizure"])

    print(f"    ✓ Created {anno_txt.name} ({pairs} seizure(s), fs={fs:.0f} Hz)")
    return True


def write_csv(file_list, csv_path: Path) -> None:
    """Write a CSV manifest with absolute paths."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename"])
        for p in file_list:
            writer.writerow([str(p.resolve())])
    print(f"  ✓ {csv_path.name}")


def partition_files(file_list: list, train_frac: float = 0.8) -> tuple:
    """Chronological 80/20 split for train/test CSVs."""
    file_list = sorted(file_list)
    split_idx = int(len(file_list) * train_frac)
    return file_list[:split_idx], file_list[split_idx:]


# ---------------------------------------------------------------------------
# Main workflows
# ---------------------------------------------------------------------------

def run_minimal(data_dir: Path, csv_dir: Path) -> bool:
    """Download the 3-file minimal subset (~120 MB)."""
    print("\n[Minimal mode] Downloading 3 files from patient chb01 …\n")
    ok = True
    all_minimal = [(p, f) for p, flist in MINIMAL_FILES.items() for f in flist]
    for patient, fname in tqdm(all_minimal, desc="Downloading EDFs", unit="file", ncols=80):
        rel = f"{patient}/{fname}"
        url = f"{BASE_URL}/{quote(rel)}?download"
        dest = data_dir / rel
        if not download(url, dest):
            ok = False
        # Also try to download .seizures annotation
        ann_url = f"{BASE_URL}/{quote(rel)}.seizures?download"
        ann_dest = data_dir / f"{rel}.seizures"
        download(ann_url, ann_dest, show_progress=False)  # May 404 — that's fine

    if not ok:
        print("\nSome downloads failed.")
        return False

    # Convert annotations
    print("\nConverting annotations …")
    for patient, fname in tqdm(all_minimal, desc="Converting", unit="file", ncols=80):
        convert_annotations(data_dir / patient / fname)

    # Create CSVs
    print("\nCreating CSV manifests …")
    all_files = [data_dir / f"{p}/{f}" for p, flist in MINIMAL_FILES.items() for f in flist]
    write_csv(all_files, csv_dir / "chb01.csv")
    write_csv(all_files, csv_dir / "chb01_test.csv")

    return True


def run_full(data_dir: Path, csv_dir: Path, patients: list, yes: bool = False) -> bool:
    """Download the full CHB-MIT dataset (~40–45 GB)."""
    print(f"\n[Full mode] Preparing to download ~{FULL_SIZE_GB} GB from PhysioNet …")
    print(f"  Patients: {', '.join(patients)}")
    print(f"  Data dir: {data_dir}")
    print(f"  CSV dir:  {csv_dir}")
    print()

    if not yes:
        ans = input("Continue? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return False

    # Discover all EDF files
    all_records = fetch_records_list()

    # Filter to requested patients
    records = [r for r in all_records if r.split("/")[0] in patients]
    if not records:
        print("No matching records found.")
        return False

    total_files = len(records)
    print(f"\nRecords to download: {total_files} EDF files")
    print()

    # Download EDFs + .seizures
    ok = True
    start_time = time.time()
    for rel in tqdm(records, desc="Downloading EDFs", unit="file", ncols=80):
        patient = rel.split("/")[0]
        fname = rel.split("/")[1]
        dest = data_dir / rel

        url = f"{BASE_URL}/{quote(rel)}?download"
        if not download(url, dest):
            ok = False
            continue

        # Annotation binary (small, no progress bar needed)
        ann_url = f"{BASE_URL}/{quote(rel)}.seizures?download"
        ann_dest = data_dir / f"{rel}.seizures"
        download(ann_url, ann_dest, show_progress=False)  # May 404 for interictal files — silent

    elapsed = time.time() - start_time
    print(f"\nDownload phase complete in {elapsed / 60:.1f} minutes.")

    if not ok:
        print("WARNING: Some files failed to download. Check output above.")

    # Convert annotations
    print("\nConverting annotations to text format …")
    for rel in tqdm(records, desc="Converting annotations", unit="file", ncols=80):
        convert_annotations(data_dir / rel)

    # Create per-patient CSVs
    print("\nCreating CSV manifests …")
    patient_files = {}
    for rel in records:
        patient = rel.split("/")[0]
        patient_files.setdefault(patient, []).append(data_dir / rel)

    all_train = []
    all_test = []

    for patient in sorted(patient_files):
        files = patient_files[patient]
        train_files, test_files = partition_files(files)

        write_csv(train_files, csv_dir / f"{patient}.csv")
        write_csv(test_files, csv_dir / f"{patient}_test.csv")

        all_train.extend(train_files)
        all_test.extend(test_files)

    # Combined multi-patient CSVs
    if len(patients) > 1:
        write_csv(all_train, csv_dir / "all_patients_train.csv")
        write_csv(all_test, csv_dir / "all_patients_test.csv")
        print(f"\n  ✓ Combined: all_patients_train.csv ({len(all_train)} files)")
        print(f"  ✓ Combined: all_patients_test.csv  ({len(all_test)} files)")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CHB-MIT Dataset Setup")
    parser.add_argument(
        "--mode", choices=["minimal", "full"], default=None,
        help="Download mode. Omit for interactive prompt."
    )
    parser.add_argument(
        "--patients", default=None,
        help="Comma-separated patient IDs (e.g., chb01,chb02). Only for --mode=full."
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR_DEFAULT, help="Output data directory"
    )
    parser.add_argument(
        "--csv-dir", default=CSV_DIR_DEFAULT, help="Output CSV directory"
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompt (useful for batch mode)."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_dir = Path(args.csv_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Interactive or CLI mode selection
    # -----------------------------------------------------------------------
    mode = args.mode
    if not mode:
        print("=" * 65)
        print("CHB-MIT Dataset Setup")
        print("=" * 65)
        print()
        print("Choose download mode:")
        print("  1) Minimal  – 3 files (~120 MB)  – smoke test only")
        print(f"  2) Full     – all patients (~{FULL_SIZE_GB} GB) – real training")
        print()
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            mode = "minimal"
        elif choice == "2":
            mode = "full"
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
        print()

    # -----------------------------------------------------------------------
    # Resolve patient list
    # -----------------------------------------------------------------------
    if mode == "minimal":
        patients = ["chb01"]
    else:
        if args.patients:
            patients = [p.strip() for p in args.patients.split(",")]
            invalid = [p for p in patients if p not in ALL_PATIENTS]
            if invalid:
                print(f"Unknown patients: {', '.join(invalid)}")
                print(f"Valid patients: {', '.join(ALL_PATIENTS)}")
                sys.exit(1)
        else:
            patients = ALL_PATIENTS

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    if mode == "minimal":
        success = run_minimal(data_dir, csv_dir)
    else:
        success = run_full(data_dir, csv_dir, patients, yes=args.yes)

    if not success:
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Done! Summary")
    print("=" * 65)
    print(f"\nData folder:  {data_dir}")
    print(f"CSV folder:   {csv_dir}")
    print("\nNext steps:")
    print("  1. Activate the venv:   source venv/bin/activate")
    print("  2. Train (1 patient):   python scrTrainLSTM.py \\")
    print("       -csv ./DataCSVs/CHB-MIT/chb01.csv \\")
    print("       -tcsv ./DataCSVs/CHB-MIT/chb01_test.csv \\")
    print("       -rf 128 -du 5 -bs 16 -hd 256 -nl 2 -os 3 \\")
    print("       -lr 0.001 -ep 20 -nw 4")
    print("  3. Or run the helper:   bash runTrainLSTM.sh")
    print()


if __name__ == "__main__":
    main()
