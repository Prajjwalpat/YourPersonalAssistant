$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeDir = Join-Path $PSScriptRoot '.runtime'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Dependencies are missing. Run: py -3.10 -m uv sync --python 3.10'
}
$ollamaPath = Join-Path $PSScriptRoot '.runtime\ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaPath)) {
    $ollamaPath = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
}
if (-not (Test-Path -LiteralPath $ollamaPath)) {
    $ollamaPath = (Get-Command ollama -ErrorAction Stop).Source
}
try {
    Invoke-RestMethod 'http://localhost:11434/api/version' -TimeoutSec 3 | Out-Null
} catch {
    $env:OLLAMA_MODELS = Join-Path $PSScriptRoot '.ollama\models'
    $env:OLLAMA_HOST = '127.0.0.1:11434'
    Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runtimeDir 'ollama.stdout.log') `
        -RedirectStandardError (Join-Path $runtimeDir 'ollama.stderr.log') | Out-Null
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod 'http://localhost:11434/api/version' -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch {}
    }
    if (-not $ready) { throw 'Ollama did not start. See .runtime/ollama.stderr.log.' }
}
try {
    $health = Invoke-WebRequest 'http://localhost:8501/_stcore/health' -UseBasicParsing -TimeoutSec 3
    if ($health.Content.Trim() -eq 'ok') {
        Write-Output 'Streamlit is already running at http://localhost:8501'
        exit 0
    }
} catch {}
$app = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'streamlit', 'run', 'src/app.py', '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8501', '--browser.gatherUsageStats=false') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDir 'streamlit.stdout.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'streamlit.stderr.log')
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if ($app.HasExited) { throw 'Streamlit stopped. See .runtime/streamlit.stderr.log.' }
    try {
        $health = Invoke-WebRequest 'http://localhost:8501/_stcore/health' -UseBasicParsing -TimeoutSec 2
        if ($health.Content.Trim() -eq 'ok') {
            Write-Output "ChatDoc is running at http://localhost:8501 (PID $($app.Id))"
            exit 0
        }
    } catch {}
}
throw 'Streamlit did not become ready. See .runtime/streamlit.stderr.log.'
