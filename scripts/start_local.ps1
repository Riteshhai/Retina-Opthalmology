param([switch]$ApiOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements first; see README.md.'
}
if (-not $ApiOnly) {
    Start-Process -FilePath $pythonPath -ArgumentList @('-m','streamlit','run','dashboard/app.py',
        '--server.address=127.0.0.1','--server.port=8501','--server.headless=true') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden
}
& $pythonPath -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
