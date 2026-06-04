#!/usr/bin/env bash

# bash runTrainLSTM.sh
#
# Improved seizure prediction with:
# - 30-second windows (literature standard, better temporal context)
# - 256 Hz sampling (full resolution, no info loss)
# - Bandpass filtering 0.5-45 Hz (removes noise)
# - Z-score normalization (per-channel)
# - Bidirectional LSTM + attention (better temporal modeling)
# - File-based train/val/test split (no temporal leakage)
# - WeightedRandomSampler only (no double class balancing)

python scrTrainLSTM.py \
  -csv './DataCSVs/CHB-MIT/all_patients_train.csv' \
  -tcsv './DataCSVs/CHB-MIT/all_patients_test.csv' \
  -rf 256 -du 30 -bs 8 -smod 1 -smin -1 -smax 1 \
  -pd 1800 -ph 300 \
  -vf 0.2 -tf 0.1 -gpu 0 -nw 4 \
  -hd 256 -nl 2 -os 3 -dr 0.5 \
  -opt 1 -lr 0.001 -ep 20 -ve 10
