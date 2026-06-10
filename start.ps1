# SpudGuard — kill old servers and start fresh
$root = $PSScriptRoot

function Stop-Port($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Write-Host "Stopping process $($c.OwningProcess) on port $port"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Freeing ports 8000 and 3000..."
Stop-Port 8000
Stop-Port 3000
Start-Sleep -Seconds 2

Set-Location $root
Write-Host "Starting SpudGuard (API + Frontend)..."
npm start
