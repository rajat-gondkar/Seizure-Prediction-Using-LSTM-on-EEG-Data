#!/usr/bin/env bash
# Run these commands on the cloud PC to collect dataset statistics

echo "=== 1. List all patients ==="
ls -d /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Data/CHB-MIT/chb*/

echo ""
echo "=== 2. Total EDF files ==="
find /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Data/CHB-MIT/ -name "*.edf" | wc -l

echo ""
echo "=== 3. Per-patient file and seizure counts ==="
for d in /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Data/CHB-MIT/chb*/; do
    pt=$(basename "$d")
    files=$(ls "$d"/*.edf 2>/dev/null | wc -l)
    seizures=$(ls "$d"/*.annotation.txt 2>/dev/null | wc -l)
    echo "  $pt: $files files, $seizures files with annotations"
done

echo ""
echo "=== 4. Training CSV class distribution (from training log) ==="
grep "Class counts in training set" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -1

echo ""
echo "=== 5. Training CSV total windows ==="
grep "Total windows in dataset" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -1

echo ""
echo "=== 6. File-based split info ==="
grep -E "Found.*unique source files|File-based split|Training set:|Validation set:|Test set:" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -15

echo ""
echo "=== 7. Class distribution details ==="
grep -A 4 "Class distribution:" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -8

echo ""
echo "=== 8. Test results ==="
grep -E "Overall accuracy|precision.*recall" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTestLSTM_*.log | tail -6

echo ""
echo "=== 9. Total training time ==="
grep "datTrainingDuration" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -1

echo ""
echo "=== 10. Model parameters ==="
grep "intTotalParams" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -1

echo ""
echo "=== 11. Training command used ==="
head -1 /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -1

echo ""
echo "=== 12. Epoch-wise losses ==="
grep -E "Epoch.*TrainLoss" /workspace/Seizure-Prediction-Using-LSTM-on-EEG-Data/Logs/runTrainLSTM_*.log | tail -25
