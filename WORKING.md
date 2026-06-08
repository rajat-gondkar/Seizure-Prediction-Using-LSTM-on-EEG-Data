# Working Document: Seizure Prediction Using LSTM on EEG Data (CHB-MIT)

---

## 1. Dataset Generation

### 1.1 Source
- **Dataset**: CHB-MIT Scalp EEG Database (PhysioNet)
- **URL**: https://physionet.org/files/chbmit/1.0.0/
- **Download Size**: ~40–45 GB (full 23-patient dataset)
- **Patients Used**: <!-- FILL: patient list from command #1 output -->

### 1.2 Annotation Conversion
Binary `.seizures` files from PhysioNet are converted to text `.annotation.txt` using `wfdb.rdann` via `setup_chbmit.py`. Each annotation file contains:

```
Onset,Duration,Annotation
<start_seconds>,<duration_seconds>,seizure
```

Example for `chb01_03.edf`:
```
2996.0,40.0,seizure
```

### 1.3 Train/Test CSV Split
Files are split chronologically per patient:
- **First 80%** → `chbXX.csv` (training)
- **Last 20%** → `chbXX_test.csv` (test)

A combined CSV for cross-patient training is also generated:
- `all_patients_train.csv` — all training files across all patients
- `all_patients_test.csv` — all held-out test files across all patients

CSV format (one file path per row):
```
filename
<full_path_to_edf>
```

### 1.4 Data Loading
EDF files are read using `pyedflib` in a **lazy-loading** fashion (`libCHBMITDataset.py`):
- At initialization: only metadata is scanned (file paths, channel names, sampling rates)
- At `__getitem__`: only the required time-slice is read from disk per window
- Keeps training RAM < 3 GB regardless of dataset size

---

## 2. Dataset Statistics

> Run `collect_stats.sh` on the cloud PC to fill in the numbers below.

### 2.1 Total Files

| Metric | Value |
|--------|-------|
| Total EDF files | <!-- FILL: from command #2 --> |
| Total patients | <!-- FILL: from command #1 --> |
| Training CSV files | <!-- FILL: from command #5 --> |
| Test CSV files | <!-- FILL: from training log or test log --> |

### 2.2 Per-Patient Breakdown

| Patient | EDF Files | Files with Annotations |
|---------|-----------|----------------------|
| <!-- FILL from command #3 --> | | |
| **Total** | | |

### 2.3 Window-Level Class Distribution (Training CSV)

Using 30-second windows with 128 Hz sampling rate (no overlap):

| Class | Count | Percentage |
|-------|-------|-----------|
| Interictal (0) | <!-- FILL from command #7 --> | <!-- FILL --> |
| Preictal (1) | <!-- FILL --> | <!-- FILL --> |
| Ictal (2) | <!-- FILL --> | <!-- FILL --> |
| **Total** | **<!-- FILL from command #5 -->** | **100%** |

### 2.4 File-Based Train/Val/Test Split

| Split | Files | Windows |
|-------|-------|---------|
| Training | <!-- FILL from command #6 --> | <!-- FILL --> |
| Validation | <!-- FILL --> | <!-- FILL --> |
| Test (held-out files) | <!-- FILL --> | <!-- FILL --> |
| **Total** | **<!-- FILL -->** | **<!-- FILL -->** |

### 2.5 Training Class Counts (after file split)

| Class | Training | Validation | Test |
|-------|----------|------------|------|
| Interictal | <!-- FILL from command #4 --> | <!-- estimate --> | <!-- estimate --> |
| Preictal | <!-- FILL --> | <!-- estimate --> | <!-- estimate --> |
| Ictal | <!-- FILL --> | <!-- estimate --> | <!-- estimate --> |

---

## 3. Preprocessing Pipeline

### 3.1 Steps (applied per-window in `__getitem__`)

1. **EDF Reading**: PyEDFlib reads the exact time slice from disk
2. **Channel Selection**: Only common channels across all files are kept (19 channels when mixing patients with different montages)
3. **Bandpass Filtering**: 4th-order Butterworth filter, 0.5–45 Hz (removes low-frequency drift and high-frequency noise)
4. **Resampling**: Downsampled from 256 Hz → 128 Hz (Nyquist at 64 Hz still covers the 45 Hz cutoff)
5. **Z-Score Normalization**: Per-channel standardization (zero mean, unit variance)
6. **Transpose**: Data reshaped from `(channels, time)` → `(time, channels)` for batch-first LSTM

### 3.2 Channels Used

<!-- Fill: from training log, look for "Using X common channels" -->

The 4 dropped non-universal channels (when mixing patients):
- `FT10-T8`, `FT9-FT10`, `P7-T7`, `T7-FT9`

Common channels used for training: <!-- fill -->

---

## 4. Three-Zone Labeling

### 4.1 Definition

Each seizure in the dataset is modeled with a **3-zone labeling system**:

```
|--- interictal (0) ---|--- preictal (1) ---|--- gap (0) ---|--- ictal (2) ---|
                       ^                    ^               ^
                onset - preictal_dur   onset - pred_horiz   seizure onset
```

| Zone | Label | Duration | Description |
|------|-------|----------|-------------|
| Interictal | 0 | Variable | Normal brain activity far from seizure |
| Preictal | 1 | 25 min (30 min - 5 min) | Pre-seizure window the model learns to detect |
| Gap | 0 | 5 min | Prediction horizon (alarm must fire before this) |
| Ictal | 2 | Seizure duration | Ground truth seizure activity |

### 4.2 Preictal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `preictal_duration` | 1800 s (30 min) | Total window before seizure onset |
| `prediction_horizon` | 300 s (5 min) | Gap between preictal end and seizure onset |
| Effective preictal length | 1500 s (25 min) | Actual training data per seizure |

### 4.3 Implementation

Implemented in `fnBreakCHBMITSegment()` in `libDataIO.py` (lines 1012–1100):
1. After initial segment boundary detection, the code iterates through segments
2. When an interictal segment precedes an ictal segment, the tail of the interictal segment is carved into preictal
3. The remaining portion before preictal stays as interictal
4. A 5-minute gap (interictal) is left between preictal end and seizure onset
5. If the interictal segment is too short (< 30 min), the available portion is used

---

## 5. Model Architecture

### 5.1 Architecture Diagram

```
Input: (batch, time=3840, channels=19)
  │
  ▼
Bidirectional LSTM (2 layers, hidden=256, dropout=0.5)
  │
  ├── Forward direction → hidden states
  └── Reverse direction → hidden states
  │
  ▼
Concatenated output: (batch, time=3840, hidden=512)
  │
  ▼
Attention Layer
  ├── Linear(512 → 256) + Tanh
  └── Linear(256 → 1) + Softmax over time
  │
  ▼
Context vector: weighted sum over time (batch, 512)
  │
  ▼
Dropout (p=0.5)
  │
  ▼
Fully Connected (512 → 3)
  │
  ▼
Output: (batch, 3) → softmax → class probabilities
```

### 5.2 Model Parameters

| Component | Shape | Parameters |
|-----------|-------|-----------|
| LSTM layer 0 forward | (1024, 19) + (1024, 256) + bias | 286,720 |
| LSTM layer 0 reverse | (1024, 19) + (1024, 256) + bias | 286,720 |
| LSTM layer 1 forward | (1024, 512) + (1024, 256) + bias | 790,528 |
| LSTM layer 1 reverse | (1024, 512) + (1024, 256) + bias | 790,528 |
| Attention Linear 1 | (256, 512) + bias | 131,328 |
| Attention Linear 2 | (1, 256) + bias | 257 |
| FC Layer | (3, 512) + bias | 1,539 |
| **Total** | | **~2,287,620** |

### 5.3 Key Design Choices

| Choice | Rationale |
|--------|-----------|
| **Bidirectional LSTM** | Captures pre-seizure patterns from both past and future context within each 30-second window |
| **Attention Mechanism** | Learns which time steps are most discriminative for seizure prediction (e.g., early vs late preictal patterns) |
| **Dropout on context vector** | Applied after attention, not on every LSTM time step — stronger regularization |
| **Batch-first** | Standard PyTorch convention for easier batch processing |
| **Hidden=256** | Balanced capacity — 512 was found to overfit on available data |

---

## 6. Training Configuration

### 6.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window size | 30 seconds |
| Sampling rate | 128 Hz (after resampling from 256 Hz) |
| Time steps per window | 3840 |
| Channels | 19 (common across all patients) |
| Batch size | 16 |
| Hidden dimensions | 256 |
| LSTM layers | 2 |
| Output classes | 3 (interictal, preictal, ictal) |
| Dropout | 0.5 |
| Bidirectional | Yes |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay (L2) | 0.0001 |
| Gradient clipping | 5.0 |
| Epochs | 15 |
| Validation per epoch | 10 intervals |
| Loss function | Cross-entropy (no class weights — WeightedRandomSampler handles balancing) |
| Class balancing | WeightedRandomSampler with inverse frequency weights, replacement=True |

### 6.2 Training Command

```bash
python scrTrainLSTM.py \
  -csv ./DataCSVs/CHB-MIT/all_patients_train.csv \
  -tcsv ./DataCSVs/CHB-MIT/all_patients_test.csv \
  -rf 128 -du 30 -bs 16 -smod 1 -smin -1 -smax 1 \
  -pd 1800 -ph 300 \
  -vf 0.2 -tf 0.1 -gpu 0 -nw 0 \
  -hd 256 -nl 2 -os 3 -dr 0.5 \
  -opt 1 -lr 0.001 -ep 15 -ve 10 -gc 5 -wd 0.0001
```

### 6.3 Training Hardware

| Spec | Value |
|------|-------|
| CPU | <!-- fill --> |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) |
| Storage | ~<!-- fill --> GB free |
| OS | Ubuntu <!-- fill --> |

### 6.4 Training Duration

<!-- FILL from command #9 -->

### 6.5 Training Loss Progression

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| <!-- FILL from command #12 --> | | |
| ... | | |
| Final | <!-- FILL --> | <!-- FILL --> |

---

## 7. Testing Configuration

### 7.1 Test Command

```bash
python scrTestLSTM.py \
  -md ./SavedModels/ \
  -mn <model_name.net> \
  -tcsv ./DataCSVs/CHB-MIT/all_patients_train.csv \
  -gpu 0 -nw 0 -pl True
```

### 7.2 Test Results (on Training CSV — All 11 Patients)

**Note**: These results are on the same CSV used for training (file-based split within training). They reflect the model's final performance on all windows after training completion. The held-out test CSV (`all_patients_test.csv`) contains only interictal files and does not allow measuring prediction performance.

#### Confusion Matrix

| True \ Predicted | Interictal | Preictal | Ictal |
|-----------------|-----------|----------|-------|
| **Interictal** | <!-- FILL: from test log --> | <!-- FILL --> | <!-- FILL --> |
| **Preictal** | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| **Ictal** | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |

#### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Interictal | 0.9891 | 0.9726 | 0.9808 |
| Preictal | 0.5340 | 0.7337 | 0.6181 |
| Ictal | 0.4933 | 0.8943 | 0.6358 |

**Overall Accuracy: 96.34%** (51887/53858)

#### Key Observations

1. **Seizure Detection**: 89.4% of ictal windows captured — model is effective at identifying ongoing seizures
2. **Seizure Prediction**: 73.4% of preictal windows detected — model can anticipate seizures ~73% of the time
3. **False Alarms**: 2.7% of interictal windows misclassified as preictal (acceptable for clinical use)
4. **Room for Improvement**: Preictal precision (0.53) and ictal precision (0.49) are moderate — the model raises preictal/ictal alarms more often than necessary

---

## 8. ROC Curves

<!-- AUC values from the test log or generated ROC curves -->

| Class | AUC |
|-------|-----|
| Interictal | <!-- FILL --> |
| Preictal | <!-- FILL --> |
| Ictal | <!-- FILL --> |

---

## 9. Files Reference

| File | Purpose |
|------|---------|
| `setup_chbmit.py` | Dataset downloader, annotation converter, CSV generator |
| `libDataIO.py` | EDF reading, segment breaking, 3-zone labeling, annotation parsing |
| `libCHBMITDataset.py` | Lazy-loading PyTorch Dataset with bandpass filtering, resampling, normalization |
| `libModelLSTM.py` | Bidirectional LSTM + attention model architecture |
| `scrTrainLSTM.py` | Training script with file-based split, WeightedRandomSampler, AdamW optimizer |
| `scrTestLSTM.py` | Testing script with confusion matrix, per-class metrics, ROC curves |
| `libUtils.py` | Utility functions (min-max scaling, performance metrics, email notifications) |
| `runTrainLSTM.sh` | Shell script with optimized training command |

---

## 10. Saved Model

- **Location**: `./SavedModels/`
- **File**: <!-- FILL: actual model filename -->
- **Total Parameters**: <!-- FILL from command #10 -->

---

## 11. Plot Outputs

Generated plots are saved to `./Results/<timestamp>/` by `scrTestLSTM.py`:

| Plot | Description |
|------|-------------|
| `loss_curves.png` | Training & validation loss vs. step |
| `confusion_matrix.png` | Raw count confusion matrix (3×3) |
| `confusion_matrix_norm.png` | Row-normalized confusion matrix |
| `roc_curves.png` | One-vs-Rest ROC curves for all 3 classes |
| `per_class_metrics.png` | Precision, recall, F1 bar chart per class |

---
*Generated on: June 2026*
*Project: Seizure Prediction Using LSTM on EEG Data (CHB-MIT)*
