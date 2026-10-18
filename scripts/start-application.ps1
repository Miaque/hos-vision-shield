param(
    [ValidateRange(1, 16)][int]$FileWorkers = 1,
    [ValidateRange(1, 16)][int]$StreamWorkers = 1,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonBinary = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonBinary)) { throw 'Run uv sync --frozen first' }
$runtimeDirectory = Join-Path $projectRoot 'data/application'
New-Item -ItemType Directory -Force $runtimeDirectory | Out-Null
$children = @()

function Start-Role([string]$Name, [string[]]$Arguments) {
    Start-Process -FilePath $pythonBinary -ArgumentList $Arguments -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDirectory "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeDirectory "$Name.stderr.log")
}

try {
    $children += Start-Role 'api' @('-m','uvicorn','vision_shield.main:app','--host','127.0.0.1','--port','8000','--no-access-log','--limit-concurrency','32')
    $children += Start-Role 'reconciler' @('-m','vision_shield.jobs.reconcile')
    for ($index = 0; $index -lt $FileWorkers; $index++) {
        $children += Start-Role "file-$index" @('-m','vision_shield.jobs.worker','files')
    }
    for ($index = 0; $index -lt $StreamWorkers; $index++) {
        $children += Start-Role "stream-$index" @('-m','vision_shield.jobs.worker','streams')
    }
    if ($Check) {
        $ready = $false
        for ($attempt = 0; $attempt -lt 100; $attempt++) {
            if (@($children | Where-Object HasExited).Count -gt 0) { throw 'Application process exited. Check data/application logs.' }
            try {
                Invoke-RestMethod 'http://127.0.0.1:8000/health/ready' -TimeoutSec 1 | Out-Null
                $ready = $true
                break
            }
            catch { Start-Sleep -Milliseconds 300 }
        }
        if (-not $ready) { throw 'Application did not become ready' }
        Write-Output 'Native API, file workers, stream workers and media service verified.'
        return
    }
    Write-Output 'API: http://127.0.0.1:8000/docs. Keep this terminal open; Ctrl+C stops these processes.'
    while ($true) {
        foreach ($child in $children) {
            if ($child.HasExited) { throw 'Application process exited. Check data/application logs.' }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($child in $children) {
        if (-not $child.HasExited) {
            # Windows venv launchers have child interpreters; revoke only this owned tree.
            & taskkill.exe /PID $child.Id /T /F 2>$null | Out-Null
            $child.WaitForExit()
        }
        $child.Dispose()
    }
}
