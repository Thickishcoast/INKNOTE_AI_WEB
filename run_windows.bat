@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo InkNote virtual environment was not found.
  echo Create it and install requirements before running this launcher.
  exit /b 1
)
if not exist ".env" copy /Y ".env.example" ".env" >nul
call .venv\Scripts\activate
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
endlocal
