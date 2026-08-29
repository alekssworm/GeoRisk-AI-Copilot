param(
    [string]$PythonPath = "",
    [switch]$Install,
    [switch]$Production,
    [switch]$OpenBrowser,
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$georiskRoot = $PSScriptRoot
$georiskCandidates = @()
if ($PythonPath) {
    $georiskCandidates += $PythonPath
}
if ($env:VIRTUAL_ENV) {
    $georiskCandidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}
$georiskCandidates += (Join-Path $georiskRoot ".venv\Scripts\python.exe")
$georiskDriveRoot = (Get-Item -LiteralPath $georiskRoot).PSDrive.Root
$georiskCandidates += (Join-Path $georiskDriveRoot ".venv\Scripts\python.exe")
$georiskPathPython = Get-Command python -ErrorAction SilentlyContinue
if ($georiskPathPython) {
    $georiskCandidates += $georiskPathPython.Source
}
$georiskPython = $georiskCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
    Select-Object -First 1
if (-not $georiskPython) {
    throw "Python was not found. Pass -PythonPath or create .venv."
}
$georiskPython = (Resolve-Path -LiteralPath $georiskPython).Path

Set-Location -LiteralPath $georiskRoot
if ($Install) {
    & $georiskPython -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

& $georiskPython -c "import fastapi, multipart, streamlit, reportlab, docx, pydeck"
if ($LASTEXITCODE -ne 0) {
    throw "Dependencies are incomplete. Re-run with -Install."
}

$env:GEORISK_ENV = if ($Production) { "production" } else { "development" }
$env:GEORISK_API_URL = "http://127.0.0.1:$ApiPort"
$env:GEORISK_FRONTEND_URL = "http://127.0.0.1:$UiPort"
$georiskRunDir = Join-Path $georiskRoot ".run"
New-Item -ItemType Directory -Force -Path $georiskRunDir | Out-Null

foreach ($georiskService in @("api", "frontend")) {
    $georiskPidPath = Join-Path $georiskRunDir "$georiskService.pid"
    if (Test-Path -LiteralPath $georiskPidPath) {
        $georiskExistingPid = Get-Content -LiteralPath $georiskPidPath -ErrorAction SilentlyContinue
        if ($georiskExistingPid -and (Get-Process -Id $georiskExistingPid -ErrorAction SilentlyContinue)) {
            throw "$georiskService is already running with PID $georiskExistingPid. Run .\stop.ps1 first."
        }
    }
}

$georiskApiArguments = @(
    "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort"
)
if (-not $Production) {
    $georiskApiArguments += "--reload"
}
$georiskApiProcess = Start-Process -FilePath $georiskPython `
    -ArgumentList $georiskApiArguments `
    -WorkingDirectory $georiskRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $georiskRunDir "api.log") `
    -RedirectStandardError (Join-Path $georiskRunDir "api.error.log") `
    -PassThru
$georiskApiProcess.Id | Set-Content -LiteralPath (Join-Path $georiskRunDir "api.pid")

$georiskUiProcess = Start-Process -FilePath $georiskPython `
    -ArgumentList @(
        "-m", "streamlit", "run", "frontend/streamlit_app.py",
        "--server.address", "127.0.0.1", "--server.port", "$UiPort"
    ) `
    -WorkingDirectory $georiskRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $georiskRunDir "frontend.log") `
    -RedirectStandardError (Join-Path $georiskRunDir "frontend.error.log") `
    -PassThru
$georiskUiProcess.Id | Set-Content -LiteralPath (Join-Path $georiskRunDir "frontend.pid")

$georiskApiReady = $false
$georiskUiReady = $false
for ($georiskAttempt = 0; $georiskAttempt -lt 30; $georiskAttempt++) {
    if (-not $georiskApiReady) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 |
                Out-Null
            $georiskApiReady = $true
        }
        catch { }
    }
    if (-not $georiskUiReady) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$UiPort" -TimeoutSec 2 |
                Out-Null
            $georiskUiReady = $true
        }
        catch { }
    }
    if ($georiskApiReady -and $georiskUiReady) { break }
    if ($georiskApiProcess.HasExited -or $georiskUiProcess.HasExited) { break }
    Start-Sleep -Seconds 1
}
if (-not $georiskApiReady -or -not $georiskUiReady) {
    throw "A service did not become ready. See .run/*.error.log."
}

Write-Host "GeoRisk started"
Write-Host "Frontend: http://127.0.0.1:$UiPort"
Write-Host "API docs: http://127.0.0.1:$ApiPort/docs"
Write-Host "Logs: $georiskRunDir"
if ($OpenBrowser) {
    Start-Process "http://127.0.0.1:$UiPort"
}
