param(
    [Parameter(Mandatory = $true)][string]$RedisExecutable,
    [Parameter(Mandatory = $true)][string]$MediaMtxExecutable,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$redisBinary = (Resolve-Path -LiteralPath $RedisExecutable).Path
$mediaBinary = (Resolve-Path -LiteralPath $MediaMtxExecutable).Path
$mediaConfig = Join-Path $projectRoot 'deploy/mediamtx.yml'
$redisDirectory = Join-Path $projectRoot 'data/redis'
$mediaDirectory = Join-Path $projectRoot 'data/mediamtx'
New-Item -ItemType Directory -Force $redisDirectory, $mediaDirectory | Out-Null
$redisConfigArgument = '../../deploy/redis.conf'

# Keep process handles so cleanup can never target an unrelated service by name.
$startedProcesses = @()
try {
    $startedProcesses += Start-Process -FilePath $redisBinary -ArgumentList @('"' + $redisConfigArgument + '"') `
        -WorkingDirectory $redisDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $redisDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $redisDirectory 'stderr.log')
    $startedProcesses += Start-Process -FilePath $mediaBinary -ArgumentList @('"' + $mediaConfig + '"') `
        -WorkingDirectory $mediaDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $mediaDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $mediaDirectory 'stderr.log')
    Write-Output 'Redis: 127.0.0.1:16379; RTSP: 127.0.0.1:8554; Media API: 127.0.0.1:9997'
    Write-Output 'Keep this terminal open. Ctrl+C stops only these child processes.'
    if ($Check) {
        $ready = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            if (@($startedProcesses | Where-Object HasExited).Count -gt 0) {
                throw 'Infrastructure failed to start. Check data/ logs.'
            }
            try {
                Invoke-RestMethod 'http://127.0.0.1:9997/v3/paths/list' -TimeoutSec 1 | Out-Null
                $ready = $true
                break
            }
            catch { Start-Sleep -Milliseconds 200 }
        }
        if (-not $ready) { throw 'MediaMTX did not become ready' }
        $pythonBinary = Join-Path $projectRoot '.venv/Scripts/python.exe'
        & $pythonBinary -c "import redis; c=redis.Redis(port=16379); assert c.ping(); assert c.config_get('appendfsync')['appendfsync']=='always'; assert c.config_get('maxmemory-policy')['maxmemory-policy']=='noeviction'; print('Local infrastructure verified')"
        if ($LASTEXITCODE -ne 0) { throw 'Redis readiness check failed' }
        return
    }
    while ($true) {
        foreach ($child in $startedProcesses) {
            if ($child.HasExited) { throw "Infrastructure process exited: $($child.Id). Check data/ logs." }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($child in $startedProcesses) {
        if (-not $child.HasExited) {
            $child.Kill()
            $child.WaitForExit()
        }
        $child.Dispose()
    }
}
