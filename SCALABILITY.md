# Scalability Issues & Suggested Fixes

> **Status:** Documented for future implementation.  
> **Context:** The current pipeline was designed for small-scale prototyping (1-2 patients) and needs architectural changes before scaling to the full CHB-MIT dataset (~22 patients, ~616 EDF files, ~24 GB raw).

---

## 1. RAM Explosion: Entire Dataset Loaded Upfront

### Problem
`libDataIO.py` loads **all** EDF files into a single giant NumPy array before training begins.

| Scale | Subsequences (1 s windows) | RAM Needed |
|-------|---------------------------|------------|
| 1 patient (~28 files) | ~100,800 | ~4.5 GB |
| 10 patients (~280 files) | ~1,008,000 | ~45 GB |
| 22 patients (~616 files) | ~2,217,600 | **~97 GB** |

With ictal oversampling (see Issue 2), this can balloon to **150-350 GB**.

### Why It Happens
`fnReadCHBMITEDFFiles_SlidingWindow()` concatenates every subsequence into `arrAllData` with shape `(channels, timepts, total_subsequences)`. This array is then passed directly to PyTorch `TensorDataset`.

### Suggested Fix
Replace the preload logic with a **custom PyTorch `Dataset`** that reads EDF files **on-the-fly** inside `__getitem__`.

- Open the EDF file only when a batch is requested.
- Extract the specific window, scale it, and return `(window, label)`.
- Keep only the current batch in RAM.
- Use `DataLoader(num_workers=4+)` for parallel loading.

> **Impact:** RAM drops from ~100 GB to ~1-2 GB (batch size dependent).

---

## 2. Physical Oversampling Multiplies Memory

### Problem
The training script replicates every ictal (seizure) subsequence in memory to balance classes.

```python
arrSeizureDataRawRep = np.tile(arrSeizureDataRaw, (1, 1, intImbalFactor))
```

If the dataset has a 50:1 interictal-to-ictal ratio and `intImbalFactor = 50`, the ictal portion is duplicated 50× in RAM.

### Suggested Fix
Use PyTorch's `WeightedRandomSampler` instead of physical duplication.

```python
from torch.utils.data import WeightedRandomSampler

# Assign higher weight to minority (ictal) classes
sample_weights = [class_weights[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(dataset, sampler=sampler, batch_size=...)
```

> **Impact:** No extra RAM for oversampling. Better randomization each epoch.

---

## 3. Sliding Windows Are Pre-Generated, Not Lazy

### Problem
The sliding window logic (e.g., 1 s windows with 256-sample step) is executed **once upfront**, creating thousands of overlapping subsequences per file. These are all stored in `arrAllData`.

### Suggested Fix
Generate windows **lazily** inside the custom `Dataset.__getitem__`.

- Pre-compute only a **lookup table** of `(file_idx, start_sample, label)` tuples.
- When `__getitem__(idx)` is called, open the file, seek to `start_sample`, read `window_length` samples.
- Optionally cache recently opened files with an LRU cache.

> **Impact:** RAM usage becomes independent of step size. You can use overlapping windows freely.

---

## 4. Scaling Reads Every EDF Twice

### Problem
`fnGetCHBMITStats()` loops through all EDF files to compute per-channel min/max/mean. Then the main loader loops through them **again** to extract data.

### Suggested Fix
Pre-compute statistics **once per patient** and save to a small JSON/CSV file.

```json
{
  "chb01": {
    "chb01_01.edf": {"min": [...], "max": [...], "mean": [...]},
    ...
  }
}
```

The loader reads this metadata file instead of re-scanning EDFs.

> **Impact:** Training startup time drops from minutes to seconds.

---

## 5. Sequence Length Is Too Granular

### Problem
With 256 Hz sampling and 1-second windows, each subsequence has only 256 time points but there are **3,600 subsequences per hour** per file. This creates an enormous number of samples.

### Suggested Fix
Use **longer windows** (5-30 seconds) and/or **downsample** to 128 Hz or 64 Hz.

| Config | Time Points / Window | Windows / Hour | RAM / Hour |
|--------|---------------------|----------------|------------|
| 256 Hz, 1 s | 256 | 3,600 | ~1.6 GB |
| 128 Hz, 5 s | 640 | 720 | ~0.8 GB |
| 64 Hz, 10 s | 640 | 360 | ~0.4 GB |

Seizure prediction does not require full 256 Hz fidelity for LSTM classification.

> **Impact:** 4-10× reduction in sample count and RAM. Faster training.

---

## 6. Patient-to-Patient Channel Mismatch

### Problem
Different patients in CHB-MIT have slightly different channel montages (e.g., 23 channels vs. 24 channels, different channel names). The current code raises an exception if channels don't match across files.

### Suggested Fix
Before multi-patient training:
1. Compute the **intersection** of channels across all patients.
2. Use only the common subset (e.g., 18-20 channels).
3. Add a channel mapping step in the loader to reorder/select channels consistently.

> **Impact:** Enables true cross-patient generalization.

---

## Summary: Effort Required

| Fix | Complexity | Effort | Priority |
|-----|-----------|--------|----------|
| Custom on-the-fly `Dataset` | Medium | 1-2 days | **Critical** |
| `WeightedRandomSampler` | Low | 2-3 hours | High |
| Lazy sliding window lookup | Low-Medium | 3-4 hours | High |
| Pre-computed scaling stats | Low | 2-3 hours | Medium |
| Longer windows / downsampling | Low | 1-2 hours | Medium |
| Cross-patient channel mapping | Medium | 4-6 hours | Medium |

**Overall:** A focused 2-3 day refactor of `libDataIO.py` and `scrTrainLSTM.py` would make the pipeline scalable to the full dataset.

---

## Alternative: Use a Subset First

If you want results quickly without refactoring, train on **5-10 patients** with a cloud instance that has **64+ GB RAM**. This avoids the scalability rewrite and still gives meaningful results for proof-of-concept. See the main README for guidance on subset selection.
