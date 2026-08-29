$georiskRoot = $PSScriptRoot
$georiskRunDir = Join-Path $georiskRoot ".run"

foreach ($georiskService in @("frontend", "api")) {
    $georiskPidPath = Join-Path $georiskRunDir "$georiskService.pid"
    if (-not (Test-Path -LiteralPath $georiskPidPath)) {
        continue
    }
    $georiskProcessId = Get-Content -LiteralPath $georiskPidPath -ErrorAction SilentlyContinue
    if ($georiskProcessId -and (Get-Process -Id $georiskProcessId -ErrorAction SilentlyContinue)) {
        & taskkill.exe /PID $georiskProcessId /T /F | Out-Null
        Write-Host "Stopped $georiskService (PID $georiskProcessId)"
    }
    Remove-Item -LiteralPath $georiskPidPath -Force
}
