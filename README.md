# Seizure Prediction Using LSTM on EEG Data

An LSTM (Long Short-Term Memory) based classifier for predicting epileptic seizures from continuous, multi-channel scalp EEG recordings.

## Overview

This project trains a deep recurrent neural network to classify EEG signal windows into three brain states:
- **Interictal** — normal brain activity (label `0`)
- **Preictal** — period immediately before a seizure (label `1`)
- **Ictal** — seizure activity (label `2`)

The model was originally designed for the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) from Boston Children's Hospital, which contains recordings from 22 pediatric patients.

![Sample EEG with annotated seizure](Images/sample_seizure.png)

---

## Repository Structure

```
.
├── scrTrainLSTM.py          # Main training script (CLI or Jupyter)
├── scrTestLSTM.py           # Main testing/inference script
├── libDataIO.py             # EEG data I/O, preprocessing, sliding windows, scaling
├── libModelLSTM.py          # LSTM model definition, save/load checkpoints
├── libUtils.py              # Utilities: plotting, memory monitoring, email alerts, etc.
├── *.ipynb                  # Jupyter notebook versions of the scripts above
├── runTrainLSTM.sh          # Example bash command for training
├── run_docker.sh            # Docker launch script for GPU environments
├── setup_test_data.sh       # One-command setup: downloads a minimal test dataset
├── setup_minimal_chbmit.py  # Python downloader for a small CHB-MIT subset
├── SCALABILITY.md           # Known bottlenecks and recommended architectural fixes
└── README.md                # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Using the provided virtual environment script
bash setup_test_data.sh

# Or manually
python3 -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib pyedflib wfdb psutil pytz
```

### 2. Download Data

For a **quick smoke test** (3 small files, ~120 MB):
```bash
source venv/bin/activate
python3 setup_minimal_chbmit.py
```

For the **full CHB-MIT dataset** (~24 GB), download from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) and create CSV manifests pointing to your `.edf` files.

### 3. Train

```bash
source venv/bin/activate

python scrTrainLSTM.py \
  -csv ./DataCSVs/CHB-MIT/chb01.csv \
  -tcsv ./DataCSVs/CHB-MIT/chb01_Test.csv \
  -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 \
  -vf 0.2 -tf 0.1 -gpu 0 \
  -hd 256 -nl 2 -os 3 -dr 0.5 \
  -opt 0 -lr 0.001 -ep 10 -ve 20 -gc 5
```

**Key arguments:**
| Flag | Description |
|------|-------------|
| `-csv` | CSV manifest with training EDF file paths |
| `-tcsv` | CSV manifest with test EDF file paths (optional, used for scaling) |
| `-rf` | Resampling frequency in Hz (`-1` = use raw) |
| `-du` | Subsequence duration in seconds (`-1` = full file) |
| `-bs` | Batch size |
| `-hd` | LSTM hidden dimension |
| `-nl` | Number of LSTM layers |
| `-os` | Number of output classes (2 or 3) |
| `-dr` | Dropout probability |
| `-lr` | Learning rate |
| `-ep` | Number of epochs |
| `-gpu` | GPU device index (`-1` = CPU) |

### 4. Test / Evaluate

```bash
python scrTestLSTM.py \
  -md ./SavedModels/ \
  -mn <your_saved_model.net> \
  -tcsv ./DataCSVs/CHB-MIT/chb01_Test.csv \
  -gpu 0 -sa True
```

The test script prints accuracy, true/false positive rates, and saves annotation files for visualization in EDFbrowser.

---

## How It Works

### Data Pipeline
1. **Read EDF** — `libDataIO.py` uses `pyedflib` to read multi-channel EEG signals, durations, sampling rates, and channel names. Non-EEG channels (EKG, VNS) are automatically discarded.
2. **Annotations** — Seizure start/end times are read from `.seizures` (binary) or `.annotation.txt` (text) files. Each recording is split into interictal, preictal, and ictal segments.
3. **Sliding Window** — Long recordings are broken into overlapping subsequences (e.g., 1-second windows). A subwindow fraction determines the label when a window straddles a boundary.
4. **Scaling** — Min-max scaling is applied per-channel across the entire dataset (or per-file, configurable).
5. **Oversampling** — Ictal samples are replicated to combat extreme class imbalance (interictal >> ictal).

### Model Architecture
- **LSTM Layer** — `nn.LSTM(input_dim, hidden_dim, num_layers, dropout=..., batch_first=True)`
- **Dropout Layer** — Regularization between LSTM and FC layers
- **Fully-Connected Layer** — Maps hidden state to class logits
- **Output** — The **last time step** of the LSTM sequence is used for classification

### Training
- Loss: `CrossEntropyLoss` (with optional class weights)
- Optimizers: Adam, AdamW, or SGD
- Gradient clipping to prevent exploding gradients in RNNs
- Validation loop runs multiple times per epoch
- Model checkpoint saved to `./SavedModels/` with all hyperparameters embedded

---

## Known Limitations & Scalability

The current pipeline is excellent for prototyping with 1-3 patients but has **memory bottlenecks** when scaling to the full dataset.

- It pre-loads **all** data into RAM as a single numpy array (~100 GB for 22 patients).
- It physically replicates ictal samples for class balancing.
- It reads every EDF twice (once for stats, once for data).

A detailed breakdown of these issues and recommended fixes is documented in [`SCALABILITY.md`](SCALABILITY.md).

**If you want to train on the full dataset, a 2-3 day refactor of the data loader is strongly recommended.**

**If you want results quickly without refactoring**, train on a **5-10 patient subset** using a cloud instance with **64+ GB RAM**.

---

## Dependencies

- Python >= 3.8
- PyTorch >= 1.6 (tested up to 2.11)
- NumPy, SciPy
- pyedflib (EDF I/O)
- wfdb (PhysioNet annotation reading)
- matplotlib, psutil, pytz

---

## Citation

If you use this code or the CHB-MIT dataset, please cite:

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.

---

## License

This repository is provided for research and educational purposes. The CHB-MIT dataset is hosted by PhysioNet under its own usage terms.
