
@echo off
cd /d "%~dp0"
start "H2V Server" /min cmd /c "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8000
