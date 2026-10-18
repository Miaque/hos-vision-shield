$ErrorActionPreference = 'Stop'
$modelsDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'models'
$manifest = Get-Content -LiteralPath (Join-Path $modelsDirectory 'cpu-manifest.json') -Raw | ConvertFrom-Json
foreach ($model in $manifest.models) {
    $targetFile = Join-Path $modelsDirectory $model.path
    if ((Test-Path -LiteralPath $targetFile) -and
        (Get-FileHash -LiteralPath $targetFile -Algorithm SHA256).Hash.ToLowerInvariant() -eq $model.sha256) {
        Write-Output "Verified $($model.path)"
        continue
    }
    $temporaryFile = "$targetFile.part"
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $model.source -OutFile $temporaryFile -TimeoutSec 30
            if ((Get-FileHash -LiteralPath $temporaryFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $model.sha256) {
                throw "SHA-256 mismatch: $($model.path)"
            }
            # A failed download must never replace an existing model.
            Move-Item -LiteralPath $temporaryFile -Destination $targetFile -Force
            break
        }
        catch {
            if ($attempt -eq 2) { throw }
        }
    }
    Write-Output "Verified $($model.path)"
}
