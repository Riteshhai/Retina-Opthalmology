param([switch]$ApiOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements first; see README.md.'
}
$localEnvPath = Join-Path $projectRoot '.env'
if (Test-Path -LiteralPath $localEnvPath) {
    foreach ($line in Get-Content -LiteralPath $localEnvPath) {
        if ($line -match '^(QSART_API_PORT|QSART_DASHBOARD_PORT)=(\d+)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}
$apiPort = if ($env:QSART_API_PORT) { [int]$env:QSART_API_PORT } else { 8000 }
$dashboardPort = if ($env:QSART_DASHBOARD_PORT) { [int]$env:QSART_DASHBOARD_PORT } else { 8501 }
if ($apiPort -lt 1024 -or $apiPort -gt 65535 -or $dashboardPort -lt 1024 -or $dashboardPort -gt 65535) {
    throw 'Choose application ports between 1024 and 65535.'
}
$env:QSART_API_URL = "http://127.0.0.1:$apiPort"
$dashboardProcess = $null
if (-not $ApiOnly) {
    $dashboardProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','streamlit','run',
        'dashboard/app.py','--server.address=127.0.0.1',"--server.port=$dashboardPort",'--server.headless=true') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
}
try {
    & $pythonPath -m uvicorn api.main:app --host 127.0.0.1 --port $apiPort --no-access-log
} finally {
    if ($dashboardProcess -and -not $dashboardProcess.HasExited) {
        Stop-Process -Id $dashboardProcess.Id
    }
}
