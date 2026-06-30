@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
)

.venv\Scripts\python -c "import fastapi, torch" 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  .venv\Scripts\python -m pip install --upgrade pip --default-timeout 600 --retries 10
  .venv\Scripts\pip install -r requirements.txt --default-timeout 600 --retries 10
  .venv\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --default-timeout 600 --retries 10
  .venv\Scripts\python scripts\verify_deps.py
  if errorlevel 1 exit /b 1
)

if not exist weights\tumor_classifier.pt (
  echo Training model on public MRI dataset...
  .venv\Scripts\python scripts\setup_lab.py
  if errorlevel 1 exit /b 1
)

if not exist frontend\dist\index.html (
  echo Building React frontend...
  where npm >nul 2>&1
  if errorlevel 1 if exist "C:\Program Files\nodejs\npm.cmd" set "PATH=C:\Program Files\nodejs;%PATH%"
  cd frontend
  if not exist node_modules call npm install
  call npm run build
  if errorlevel 1 (
    echo Frontend build failed. Install Node.js 18+ and retry.
    cd ..
    pause
    exit /b 1
  )
  cd ..
)

echo.
echo Hospital MRI Assistant running at http://127.0.0.1:8781
echo Waiting for server, then opening browser...
echo Press Ctrl+C to stop.
start "" powershell -NoProfile -WindowStyle Hidden -Command "$u='http://127.0.0.1:8781'; for($i=0;$i -lt 60;$i++){ try { if((Invoke-WebRequest -Uri ($u + '/api/health') -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200){ Start-Process $u; break } } catch {}; Start-Sleep -Seconds 1 }"
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8781
