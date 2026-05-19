@echo off
setlocal
cd /d "%~dp0\.."
python src\spaceship_raw_trio_groupaware_retrain.py --threads 8 --xgb-device cuda --apply-residual-stage --strict-candidates

