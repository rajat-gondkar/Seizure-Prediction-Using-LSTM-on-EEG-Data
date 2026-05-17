#!/usr/bin/env bash

# bash runTrainLSTM.sh
#
# Cloud-optimized defaults for 16 GB RAM / 6 GB VRAM (e.g., RTX 4050)
# - 128 Hz resampling (halves memory vs 256 Hz)
# - 5-second windows (fewer, richer samples)
# - batch size 16 (fits 6 GB VRAM)
# - num_workers 4 (parallel data loading)
# - Uses WeightedRandomSampler instead of physical oversampling

python scrTrainLSTM.py \
  -csv './DataCSVs/CHB-MIT/chb01.csv' \
  -tcsv './DataCSVs/CHB-MIT/chb01_Test.csv' \
  -rf 128 -du 5 -bs 16 -smod 1 -smin -1 -smax 1 \
  -vf 0.2 -tf 0.1 -gpu 0 -nw 4 \
  -hd 256 -nl 2 -os 3 -dr 0.5 \
  -opt 0 -lr 0.001 -ep 10 -ve 20
