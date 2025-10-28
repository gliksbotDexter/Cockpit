# Quick Restart - Kills old Dexter and starts fresh
Write-Host "=== Quick Restart ===" -ForegroundColor Cyan

# Kill old processes
Write-Host "Stopping old processes..." -ForegroundColor Yellow
Get-Process -Name "DexterCockpit" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*start_backend*"} | Stop-Process -Force
Start-Sleep -Seconds 1

# Start fresh
Write-Host "Starting Dexter..." -ForegroundColor Green
.\start-dexter.ps1
