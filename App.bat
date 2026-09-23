@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Follow docs\MULTIMODAL.md to install the project environment.
  pause
  exit /b 1
)
.venv\Scripts\python.exe app.py
pause
