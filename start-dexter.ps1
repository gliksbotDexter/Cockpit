# Dexter-Gliksbot Unified Launcher
# This script starts both the Python backend and the WPF Cockpit UI

Write-Host "=== Dexter-Gliksbot Launcher ===" -ForegroundColor Cyan
Write-Host ""

# Kill any existing instances first
Write-Host "Checking for existing processes..." -ForegroundColor Yellow
$existingCockpit = Get-Process -Name "DexterCockpit" -ErrorAction SilentlyContinue
if ($existingCockpit) {
    Write-Host "Stopping existing DexterCockpit processes..." -ForegroundColor Yellow
    $existingCockpit | Stop-Process -Force
    Start-Sleep -Seconds 1
}

# Paths
$PythonExe = "C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe"
$BackendScript = ".\start_backend.py"
$CockpitExe = ".\DexterCockpit\bin\Debug\net8.0-windows\DexterCockpit.exe"

# Check if Python backend script exists
if (-not (Test-Path $BackendScript)) {
    Write-Host "ERROR: Backend script not found: $BackendScript" -ForegroundColor Red
    exit 1
}

# Clean up old cockpit folder if it exists
if (Test-Path ".\cockpit") {
    Write-Host "Removing old cockpit folder..." -ForegroundColor Yellow
    Remove-Item -Path ".\cockpit" -Recurse -Force -ErrorAction SilentlyContinue
}

# Always rebuild Cockpit to ensure latest XAML changes are compiled
Write-Host "Building Cockpit UI..." -ForegroundColor Yellow
& dotnet clean ".\DexterCockpit\DexterCockpit.csproj" | Out-Null
& dotnet build ".\DexterCockpit\DexterCockpit.csproj" -c Debug
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build Cockpit UI" -ForegroundColor Red
    exit 1
}
Write-Host "Cockpit UI build complete" -ForegroundColor Green

Write-Host "Starting Python backend..." -ForegroundColor Green
$BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList $BackendScript -PassThru -WindowStyle Normal

Write-Host "Waiting 3 seconds for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Starting Cockpit UI..." -ForegroundColor Green
$CockpitProcess = Start-Process -FilePath $CockpitExe -PassThru

Write-Host ""
Write-Host "=== Dexter-Gliksbot is running ===" -ForegroundColor Cyan
Write-Host "Backend PID: $($BackendProcess.Id)" -ForegroundColor Gray
Write-Host "Cockpit PID: $($CockpitProcess.Id)" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop both processes..." -ForegroundColor Yellow

try {
    # Wait for either process to exit
    while (-not $BackendProcess.HasExited -and -not $CockpitProcess.HasExited) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
    if (-not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -Force
        Write-Host "Backend stopped" -ForegroundColor Gray
    }
    if (-not $CockpitProcess.HasExited) {
        Stop-Process -Id $CockpitProcess.Id -Force
        Write-Host "Cockpit stopped" -ForegroundColor Gray
    }
}

Write-Host "Dexter-Gliksbot shutdown complete" -ForegroundColor Cyan
