@echo off
setlocal

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Could not switch to the project directory.
    pause
    exit /b 1
)

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo Starting Traffic Congestion Prediction API...
echo API docs: http://localhost:8000/docs
echo Press Ctrl+C to stop the server.
echo.

"%PYTHON%" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
if errorlevel 1 (
    echo.
    echo ERROR: API failed to start. Ensure Python and the project dependencies are installed.
    echo Install dependencies with: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

endlocal
