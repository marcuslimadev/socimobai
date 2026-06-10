param(
    [string]$ModelDir = "models\socimobai-distilled-qwen-v5",
    [int]$Port = 8111,
    [string]$HostName = "127.0.0.1",
    [string]$ApiKeyFile = ".tmp-v5-api-key.txt",
    [string]$NgrokExe = "tools\ngrok\ngrok.exe"
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$RelativePath) {
    return Join-Path $PSScriptRoot ".." $RelativePath | Resolve-Path
}

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path $ApiKeyFile)) {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $key = [Convert]::ToBase64String($bytes).TrimEnd("=") -replace "\+", "-" -replace "/", "_"
    Set-Content -Path $ApiKeyFile -Value $key -NoNewline
}

$apiKey = (Get-Content $ApiKeyFile -Raw).Trim()

if (-not (Test-Path $ModelDir)) {
    throw "Modelo nao encontrado: $ModelDir"
}

if (-not (Test-Path $NgrokExe)) {
    throw "ngrok nao encontrado em $NgrokExe. Baixe para tools\ngrok\ngrok.exe ou informe -NgrokExe."
}

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"

$healthUrl = "http://${HostName}:$Port/health"
$apiAlive = $false
try {
    $health = Invoke-RestMethod $healthUrl -TimeoutSec 3
    $apiAlive = [bool]$health.success
} catch {
    $apiAlive = $false
}

if (-not $apiAlive) {
    Start-Process -FilePath "python" `
        -ArgumentList @(
            "scripts\serve_finetuned_model.py",
            "--model-dir", $ModelDir,
            "--host", $HostName,
            "--port", $Port,
            "--api-key", $apiKey
        ) `
        -WorkingDirectory (Get-Location).Path `
        -WindowStyle Hidden `
        -RedirectStandardOutput ".tmp-v5-local-api.out.log" `
        -RedirectStandardError ".tmp-v5-local-api.err.log"

    Start-Sleep -Seconds 8
}

try {
    Invoke-RestMethod $healthUrl -TimeoutSec 10 | Out-Null
} catch {
    throw "API local nao respondeu em $healthUrl. Veja .tmp-v5-local-api.err.log"
}

$ngrokApiAlive = $false
try {
    $tunnels = Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3
    $ngrokApiAlive = ($tunnels.tunnels | Where-Object { $_.config.addr -like "*:$Port" }).Count -gt 0
} catch {
    $ngrokApiAlive = $false
}

if (-not $ngrokApiAlive) {
    Start-Process -FilePath $NgrokExe `
        -ArgumentList @("http", $Port) `
        -WorkingDirectory (Get-Location).Path `
        -WindowStyle Hidden `
        -RedirectStandardOutput ".tmp-v5-ngrok.out.log" `
        -RedirectStandardError ".tmp-v5-ngrok.err.log"

    Start-Sleep -Seconds 5
}

$publicUrl = $null
for ($i = 0; $i -lt 12; $i++) {
    try {
        $tunnels = Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 5
        $publicUrl = ($tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
        if ($publicUrl) { break }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $publicUrl) {
    throw "ngrok nao retornou URL publica. Veja .tmp-v5-ngrok.err.log"
}

Write-Host "SocimobAI v5 local ativo."
Write-Host "API local: $healthUrl"
Write-Host "Ngrok URL: $publicUrl"
Write-Host ""
Write-Host "No gateway online, use:"
Write-Host "LOCAL_MODEL_BASE_URL=$publicUrl"
Write-Host "LOCAL_MODEL_API_KEY=<conteudo de $ApiKeyFile>"
Write-Host "LOCAL_MODEL_NAME=socimobai-distilled-qwen-v5-ngrok"
Write-Host "LOCAL_MODEL_TIMEOUT_MS=45000"
