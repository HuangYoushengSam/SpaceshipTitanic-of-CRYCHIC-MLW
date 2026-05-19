@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo ============================================================
echo Spaceship Titanic ML Workshop - GPU run
echo ============================================================
echo [info] Working directory: %CD%

if not exist "data\train.csv" (
    echo [error] Missing data\train.csv
    echo [hint] Copy the original train.csv into the data folder, then run again.
    pause
    exit /b 1
)

if not exist "data\test.csv" (
    echo [error] Missing data\test.csv
    echo [hint] Copy the original test.csv into the data folder, then run again.
    pause
    exit /b 1
)

set "PYTHON_EXE=python"
where python >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\anaconda3\python.exe" (
        set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
    )
)

echo [info] Python command: %PYTHON_EXE%
%PYTHON_EXE% --version
if errorlevel 1 (
    echo [error] Python is not available. Please install Python or activate the Anaconda environment.
    pause
    exit /b 1
)

if not exist "outputs\logs" mkdir "outputs\logs"
set "LOG_FILE=outputs\logs\run_gpu.log"
echo [info] Console log will also be saved to %LOG_FILE%
echo [info] XGBoost will try CUDA first and fall back to CPU if CUDA is unavailable.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & '%PYTHON_EXE%' -u 'src\spaceship_raw_trio_groupaware_retrain.py' --threads 8 --xgb-device cuda --apply-residual-stage --strict-candidates 2>&1 | Tee-Object -FilePath '%LOG_FILE%'; exit $LASTEXITCODE }"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [done] Run completed successfully.
    echo [output] Check outputs\raw_trio_groupaware_retrain
) else (
    echo [error] Run failed with exit code %EXIT_CODE%.
    echo [log] Please check %LOG_FILE%
)

pause
exit /b %EXIT_CODE%
