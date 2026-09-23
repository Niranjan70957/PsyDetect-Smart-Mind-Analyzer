@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: uv venv --python 3.10 .venv
  exit /b 1
)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
pause
