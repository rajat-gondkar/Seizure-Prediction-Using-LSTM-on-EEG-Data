#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Seizure Prediction: Minimal CHB-MIT Test Data Setup ==="

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing data-pipeline dependencies..."
pip install -q --upgrade pip
pip install -q numpy scipy matplotlib pyedflib wfdb psutil pytz

echo ""
echo "Running minimal dataset download & preparation..."
python3 setup_chbmit.py --mode minimal

echo ""
echo "=== Setup complete ==="
echo ""
echo "Your virtual environment is ready at: ./venv"
echo ""
echo "To train (install PyTorch first):"
echo "  source venv/bin/activate"
echo "  pip install torch"
echo "  python scrTrainLSTM.py -csv ./DataCSVs/CHB-MIT/chb01.csv -tcsv ./DataCSVs/CHB-MIT/chb01_Test.csv -bs 2 -hd 64 -nl 1 -os 3 -lr 0.001 -ep 1 -smod 1 -smin -1 -smax 1"
echo ""
