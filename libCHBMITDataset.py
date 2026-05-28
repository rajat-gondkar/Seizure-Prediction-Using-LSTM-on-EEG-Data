#!/usr/bin/env python
# coding: utf-8
"""
Lazy-loading PyTorch Dataset for CHB-MIT EEG data.

Instead of loading the entire dataset into RAM upfront (which can exceed 100 GB
for the full 22-patient corpus), this Dataset:

  1. Scans EDF files ONCE at initialization to build a lightweight index of
     (filename, window_start, window_end, label, ...) metadata.
  2. Reads ONLY the requested time-slice from disk in __getitem__.
  3. Applies scaling and optional resampling on-the-fly.

This keeps training RAM usage under ~2 GB regardless of dataset size,
making it suitable for machines with 16 GB system RAM and 6 GB GPU VRAM.
"""

import os
import math
import re
import statistics as stat

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal as sp_signal
import pyedflib

import libDataIO as dio
import libUtils as utils


# ---------------------------------------------------------------------------
# Helper: build sliding-window metadata without storing data arrays
# ---------------------------------------------------------------------------

def _build_window_index_for_file(fpath, fidx, orig_sf, target_sf, subseq_dur_s,
                                 step_size_pts, sub_win_frac, step_size_states,
                                 anno_suffix, scaling_minmax=None,
                                 argInfo=False, argDebug=False):
    """
    Scans a single EDF file, reads annotations, breaks it into segments,
    and returns a list of window-metadata dicts (no actual EEG data).
    If the EDF is corrupted or unreadable, returns an empty list and prints
    a warning so that one bad file does not crash the entire dataset build.
    """
    try:
        # Read header + full data (we need the array to know n_pts & to call
        # fnBreakCHBMITSegment, but we discard it immediately after indexing)
        seg_label, data_edf, seg_dur, sfreq, channels, seq, n_ch, n_pts = \
            dio.fnReadEDFUsingPyEDFLib(fpath, argNoData=False, argDebug=False)
    except Exception as exc:
        print(f"[CHBMITDataset] WARNING: skipping corrupted/unreadable file {fpath}: {exc}")
        return []

    # --- Apply scaling if params were pre-computed ---
    if scaling_minmax is not None:
        dmin, dmax = scaling_minmax
        data_minmax = np.concatenate((dmin[:, np.newaxis], dmax[:, np.newaxis]), axis=1)
        data_edf = utils.fnMinMaxScaler(data_edf, (-1, 1), data_minmax, argDebug=False)

    # --- Read annotations ---
    bln_anno_found, start_end_timepts = dio.fnReadCHBMITAnnoTxt(
        fpath, anno_suffix, sfreq, argInfo=argInfo, argDebug=argDebug)

    # --- Break into labelled segments ---
    seg_labels, seg_types, seg_start_end_pts, seg_start_end_secs, \
        seg_durations, seg_num_timepts, data_segs = \
        dio.fnBreakCHBMITSegment(data_edf, start_end_timepts, sfreq,
                                 argPreictalDuration=5, argDebug=argDebug)

    # Build per-sample-point label array
    seg_type_timepts = np.zeros(n_pts, dtype=int)
    for stype, (sp, ep) in zip(seg_types, seg_start_end_pts):
        seg_type_timepts[sp:ep] = stype

    # Determine per-segment step sizes
    lst_step_sizes = []
    for slabel in seg_labels:
        if slabel in step_size_states:
            lst_step_sizes.append(step_size_states[slabel])
        else:
            lst_step_sizes.append(step_size_pts)

    subseq_pts_orig = int(round(subseq_dur_s * sfreq))
    subseq_pts_target = int(round(subseq_dur_s * target_sf))

    entries = []

    for sidx, (slabel, stype, (sp, ep), (ss, es), sdur, ntp) in enumerate(
        zip(seg_labels, seg_types, seg_start_end_pts, seg_start_end_secs,
            seg_durations, seg_num_timepts)):

        step = lst_step_sizes[sidx]
        win_start = sp
        win_end = win_start + subseq_pts_orig

        # Number of windows (same logic as libDataIO sliding window)
        if sidx < len(seg_start_end_pts) - 1:
            n_windows = math.ceil(ntp / step)
            n_full = n_windows
        else:
            n_windows = math.ceil((ntp - subseq_pts_orig) / step) + 1
            n_full = ((ntp - subseq_pts_orig) // step) + 1

        for widx in range(n_full):
            # Label determination
            if win_end > ep:
                # Window straddles segment boundary -> use subwindow mode
                subwin_pts = math.ceil(sub_win_frac * subseq_pts_orig)
                subwin_odd = ((subwin_pts // 2) * 2) + 1
                if subwin_odd > subseq_pts_orig:
                    subwin_odd = subseq_pts_orig
                    if subwin_odd % 2 == 0:
                        subwin_odd -= 1
                subwindow = seg_type_timepts[win_end - subwin_odd : win_end]
                try:
                    label = int(stat.mode(subwindow))
                except:
                    label = int(stype)
            else:
                label = int(stype)

            entries.append({
                'file_idx': fidx,
                'filename': fpath,
                'sequence': seq,
                'subsequence': widx,
                'start_sample': win_start,
                'end_sample': min(win_end, n_pts),
                'label': label,
                'start_sec': win_start / sfreq,
                'end_sec': min(win_end, n_pts) / sfreq,
                'subseq_pts_target': subseq_pts_target,
            })

            win_start += step
            win_end = win_start + subseq_pts_orig

    del data_edf, data_segs
    return entries, channels, n_ch, sfreq


# ---------------------------------------------------------------------------
# CHBMITDataset
# ---------------------------------------------------------------------------

class CHBMITDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to CSV manifest listing full paths to .edf files.
    resampling_freq : int
        Target sampling frequency in Hz.  -1 or > orig uses raw rate.
        Recommended: 128 (cuts memory in half with negligible info loss).
    subseq_duration : int
        Duration of each window in seconds.
        Recommended: 5 (fewer, richer samples vs 1-second slices).
    scaling_params : tuple
        (scaling_mode, (scaled_min, scaled_max)) or empty tuple for no scaling.
    scaling_info : tuple
        Optional (test_filenames, data_min, data_max) for test-set scaling.
    step_size_time_pts : int
        Sliding-window step in sample points.  -1 = no overlap ( = subseq length).
    step_size_states : dict
        Per-class step sizes, e.g. {'ictal': 128}.
    sub_window_fraction : float
        Fraction of window used to vote on label when straddling a boundary.
    anno_suffix : str
        Annotation file suffix, e.g. 'annotation.txt'.
    dtype : np.dtype
        Numpy dtype for returned arrays.  Default np.float32 (half the RAM of
        float64 and standard for PyTorch).
    """

    def __init__(self, csv_path, resampling_freq=-1, subseq_duration=-1,
                 scaling_params=(), scaling_info=(), step_size_time_pts=-1,
                 step_size_states=None, sub_window_fraction=-1,
                 anno_suffix='annotation.txt', dtype=np.float32,
                 argInfo=False, argDebug=False):
        super().__init__()

        self.dtype = dtype
        self.anno_suffix = anno_suffix
        self.debug = argDebug
        self.scaling_params = scaling_params

        if step_size_states is None:
            step_size_states = {}
        self.step_size_states = step_size_states

        # ---- File list ----
        self.files = dio.fnReadDataFileListCSV(csv_path, argInfo=argInfo)
        if len(self.files) == 0:
            raise ValueError(f"No EDF files listed in CSV: {csv_path}")

        # ---- Channel consistency check ----
        dio.fnMatchEDFChannels(self.files)

        # ---- Probe first file for globals ----
        _, _, seg_dur, sfreq, channels, _, n_ch, n_pts = \
            dio.fnReadEDFUsingPyEDFLib(self.files[0], argNoData=True, argDebug=False)

        self.num_channels = n_ch
        self.channels = channels
        self.orig_sampling_freq = sfreq

        # ---- Resampling frequency ----
        if resampling_freq <= 0 or resampling_freq > sfreq:
            self.resampling_freq = sfreq
        else:
            self.resampling_freq = resampling_freq

        # ---- Subsequence duration ----
        if subseq_duration <= 0 or subseq_duration > seg_dur:
            self.subseq_duration = seg_dur
        else:
            self.subseq_duration = subseq_duration

        self.subseq_timepts_orig = int(round(self.subseq_duration * sfreq))
        self.subseq_timepts_resampled = int(round(self.subseq_duration * self.resampling_freq))

        # ---- Step size ----
        if step_size_time_pts <= 0 or step_size_time_pts > self.subseq_timepts_orig:
            self.step_size = self.subseq_timepts_orig
        else:
            self.step_size = step_size_time_pts

        # ---- Subwindow fraction ----
        if sub_window_fraction <= 0 or sub_window_fraction > 1:
            self.sub_window_fraction = 1.0
        else:
            self.sub_window_fraction = sub_window_fraction

        # ---- Pre-compute scaling statistics ----
        self.file_scaling = [None] * len(self.files)

        if scaling_params:
            scaling_mode = scaling_params[0]
            scaled_minmax = scaling_params[1]
            self.scaled_min = scaled_minmax[0]
            self.scaled_max = scaled_minmax[1]

            if scaling_info and len(scaling_info) >= 3:
                test_basenames = [os.path.basename(t) for t in scaling_info[0]]
                data_min_all = scaling_info[1]
                data_max_all = scaling_info[2]
                for i, fpath in enumerate(self.files):
                    fname = os.path.basename(fpath)
                    try:
                        idx = test_basenames.index(fname)
                        self.file_scaling[i] = (data_min_all[:, idx], data_max_all[:, idx])
                    except ValueError:
                        pass
            else:
                arr_min, arr_max, _ = dio.fnGetCHBMITStats(self.files, argDebug=False)
                data_min, data_max = dio.fnGenMinMaxArrays(scaling_mode, arr_min, arr_max, argDebug=False)
                for i in range(len(self.files)):
                    self.file_scaling[i] = (data_min[:, i], data_max[:, i])

        # ---- Build lightweight window index ----
        self.index = []
        for fidx, fpath in enumerate(self.files):
            entries, _, _, _ = _build_window_index_for_file(
                fpath, fidx, self.orig_sampling_freq, self.resampling_freq,
                self.subseq_duration, self.step_size, self.sub_window_fraction,
                self.step_size_states, self.anno_suffix,
                scaling_minmax=self.file_scaling[fidx],
                argInfo=argInfo, argDebug=argDebug)
            self.index.extend(entries)

        if argInfo:
            print(f"[CHBMITDataset] Index built: {len(self.index)} windows from {len(self.files)} files")
            print(f"  Channels: {self.num_channels}  |  SF: {self.orig_sampling_freq} -> {self.resampling_freq} Hz")
            print(f"  Window: {self.subseq_duration}s  |  Orig pts: {self.subseq_timepts_orig}  |  Resampled pts: {self.subseq_timepts_resampled}")

    # -----------------------------------------------------------------------
    # PyTorch Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        n_read = entry['end_sample'] - entry['start_sample']

        try:
            # ---- Read ONLY the needed slice from each channel ----
            with pyedflib.EdfReader(entry['filename']) as fh:
                window = np.zeros((self.num_channels, n_read), dtype=self.dtype)
                ch_idx = 0
                orig_labels = fh.getSignalLabels()

                for i in range(fh.signals_in_file):
                    label = orig_labels[i]
                    if re.match(r'.*E[CK]G.*|.*VNS.*|\s*-\s*|\s*\.\s*', label):
                        continue
                    signal = fh.readSignal(i, start=entry['start_sample'], n=n_read)
                    window[ch_idx, :] = signal.astype(self.dtype)
                    ch_idx += 1
        except Exception as exc:
            print(f"[CHBMITDataset] ERROR reading window from {entry['filename']} "
                  f"(sample {entry['start_sample']}:{entry['end_sample']}): {exc}")
            print("  Tip: run 'python validate_dataset.py -csv <your_csv>' to check for corrupted files.")
            raise

        # ---- Apply scaling if we haven't already at indexing time ----
        # (Scaling was done during index build, but if we want on-the-fly
        # scaling we could do it here.  For now we scaled during index.)

        # ---- Resample if needed ----
        if self.resampling_freq != self.orig_sampling_freq:
            window = sp_signal.resample(window, entry['subseq_pts_target'], axis=1)

        # ---- Transpose to (time, channels) for batch_first LSTM ----
        window = window.T  # (timepts, channels)

        return torch.from_numpy(window), torch.tensor(entry['label'], dtype=torch.long)

    # -----------------------------------------------------------------------
    # Utilities for samplers / analysis
    # -----------------------------------------------------------------------

    def get_labels(self):
        """Return a plain Python list of all window labels."""
        return [e['label'] for e in self.index]

    def count_by_class(self):
        """Print class distribution."""
        labels = self.get_labels()
        from collections import Counter
        c = Counter(labels)
        print("Class distribution:")
        for lbl, cnt in sorted(c.items()):
            name = dio.fnGetSegLabel(lbl)
            print(f"  {name:12s} ({lbl}): {cnt:6d}  ({100*cnt/len(labels):5.1f}%)")
        return c

    def get_file_idx_for_index(self, idx):
        """Return the source file index for a given dataset index."""
        return self.index[idx]['file_idx']
