<#
Start Dev Environment (backend + frontend) helper

Usage:
  # Start both services in background (default)
  .\scripts\start_dev.ps1

  # Start backend in foreground (useful to see uvicorn logs directly)
  .\scripts\start_dev.ps1 -Foreground

  # Stop both services (reads PID files written when started)
  .\scripts\start_dev.ps1 -Stop

What it does:
 - Starts backend (uvicorn via the project's virtualenv python if present)
 - Starts frontend (npm run dev in frontend/)
 - Creates `backend/logs/` and `frontend/logs/` and writes stdout/stderr logs
 - Writes PID files for later stopping
#>
param(
    [switch]$Stop,
    [switch]$Foreground
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Resolve-Python {
    $candidates = @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')
    foreach ($c in $candidates) {
        $p = Join-Path $root $c
        if (Test-Path $p) { return $p }
    }
    return 'python'
}

$python = Resolve-Python

$backendLogDir = Join-Path $root 'backend\logs'
$frontendLogDir = Join-Path $root 'frontend\logs'
New-Item -ItemType Directory -Path $backendLogDir -Force | Out-Null
New-Item -ItemType Directory -Path $frontendLogDir -Force | Out-Null

$backendOut = Join-Path $backendLogDir 'backend_stdout.log'
$backendErr = Join-Path $backendLogDir 'backend_stderr.log'
$backendPid = Join-Path $backendLogDir 'backend.pid'

$frontendOut = Join-Path $frontendLogDir 'frontend_stdout.log'
$frontendErr = Join-Path $frontendLogDir 'frontend_stderr.log'
$frontendPid = Join-Path $frontendLogDir 'frontend.pid'

if ($Stop) {
    Write-Host "Stopping dev services..." -ForegroundColor Yellow
    if (Test-Path $backendPid) {
        try {
            $pid = Get-Content $backendPid -ErrorAction Stop | Select-Object -First 1
            if ($pid) { Stop-Process -Id ([int]$pid) -ErrorAction SilentlyContinue -Force; Write-Host "Stopped backend PID $pid" }
            Remove-Item $backendPid -ErrorAction SilentlyContinue
        } catch { Write-Warning "Could not stop backend: $_" }
    } else {
        Write-Host "No backend.pid found, skipping backend stop." -ForegroundColor DarkGray
    }

    if (Test-Path $frontendPid) {
        try {
            $pid = Get-Content $frontendPid -ErrorAction Stop | Select-Object -First 1
            if ($pid) { Stop-Process -Id ([int]$pid) -ErrorAction SilentlyContinue -Force; Write-Host "Stopped frontend PID $pid" }
            Remove-Item $frontendPid -ErrorAction SilentlyContinue
        } catch { Write-Warning "Could not stop frontend: $_" }
    } else {
        Write-Host "No frontend.pid found, skipping frontend stop." -ForegroundColor DarkGray
    }
    return
}

# Start backend
$uvicornArgs = @('-m', 'uvicorn', 'backend.src.api:app', '--host', '127.0.0.1', '--port', '8000')

if ($Foreground) {
    Write-Host "Starting backend in foreground using: $python $($uvicornArgs -join ' ')" -ForegroundColor Green
    & $python @uvicornArgs
    return
}

Write-Host "Starting backend in background (logs -> $backendOut, $backendErr)" -ForegroundColor Green
$backendProc = Start-Process -FilePath $python -ArgumentList $uvicornArgs -WorkingDirectory $root -NoNewWindow -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -PassThru
if ($backendProc) {
    $backendProc.Id | Out-File -FilePath $backendPid -Encoding ascii
    Write-Host "Backend started (PID: $($backendProc.Id))"
} else {
    Write-Warning "Failed to start backend process"
}

# Start frontend
Write-Host "Starting frontend (npm run dev) in background (logs -> $frontendOut, $frontendErr)" -ForegroundColor Green
$npmCmd = 'npm'
$npmArgs = @('run','dev')
$frontendWorkdir = Join-Path $root 'frontend'
$frontendProc = Start-Process -FilePath $npmCmd -ArgumentList $npmArgs -WorkingDirectory $frontendWorkdir -NoNewWindow -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru
if ($frontendProc) {
    $frontendProc.Id | Out-File -FilePath $frontendPid -Encoding ascii
    Write-Host "Frontend started (PID: $($frontendProc.Id))"
} else {
    Write-Warning "Failed to start frontend process"
}

Write-Host "\nTail logs (recommended):" -ForegroundColor Cyan
Write-Host "  Get-Content -Path '$backendOut' -Wait -Tail 200" -ForegroundColor DarkGray
Write-Host "  Get-Content -Path '$frontendOut' -Wait -Tail 200" -ForegroundColor DarkGray
Write-Host "\nTo stop services run: .\scripts\start_dev.ps1 -Stop" -ForegroundColor Yellow
