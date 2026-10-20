#!/bin/bash
# Double-click in Finder: sets up a virtual env, installs packages, runs the offline
# tests and executes the notebook with the real API. Log goes to outputs/run_log.txt
cd "$(dirname "$0")"
exec > >(tee outputs/run_log.txt) 2>&1
PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 "$HOME/anaconda3/bin/python" "$HOME/opt/anaconda3/bin/python" /opt/anaconda3/bin/python "$HOME/miniconda3/bin/python" python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "ERROR: need Python 3.10+"; exit 1; fi
echo "Using $PY ($($PY --version))"
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
source .venv/bin/activate
pip install -q --upgrade pip 2>/dev/null
pip install -q -r requirements.txt jupyter nbconvert ipykernel 2>/dev/null || { echo "PIP FAILED"; exit 1; }
echo "=== API check ==="
python quick_check.py || { echo "API CHECK FAILED"; echo "=== DONE ==="; exit 1; }
echo "=== Offline tests ==="
python -m pytest -q
echo "=== Running notebook with the real API ==="
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Customer_Service_Chatbot.ipynb && echo "=== NOTEBOOK OK ===" || echo "=== NOTEBOOK FAILED ==="
echo "=== DONE ==="
