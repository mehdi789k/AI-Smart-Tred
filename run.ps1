param(
    [string]$InputParquet = "",
    [string]$OutputFeatures = "data/features/sample_EURUSD_M1_features.parquet",
    [string]$OutputLabeled = "data/features/sample_EURUSD_M1_labeled.parquet",
    [switch]$SkipTests,
    [switch]$SkipLabeling,
    [switch]$SkipSampleGeneration
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Resolve-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "Python was not found on PATH. Install Python first: https://www.python.org/downloads/windows/"
}

$pyCmd = Resolve-PythonCommand

$venvPath = Join-Path $projectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[INFO] Creating virtual environment..."
    & $pyCmd -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment was not created correctly: $pythonExe"
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
}

Write-Host "[INFO] Installing dependencies..."
if (Test-Path (Join-Path $projectRoot "requirements.txt")) {
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -r (Join-Path $projectRoot "requirements.txt")
}
else {
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install pandas numpy pyarrow pandas-ta scipy duckdb pytest
}

if (-not $SkipTests) {
    Write-Host "[INFO] Running tests..."
    & $pythonExe -m pytest tests/test_feature_engine.py tests/test_target_labeling.py -q
}

if (-not $SkipSampleGeneration) {
    $samplePath = Join-Path $projectRoot "data/raw/sample_EURUSD_M1.parquet"
    if (-not (Test-Path $samplePath) -and -not $InputParquet) {
        $rawDir = Split-Path -Parent $samplePath
        New-Item -ItemType Directory -Path $rawDir -Force | Out-Null
        Write-Host "[INFO] Creating synthetic sample Parquet..."
        @'
import pandas as pd
import numpy as np
from pathlib import Path
Path('data/raw').mkdir(parents=True, exist_ok=True)
rng = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='T')
open_ = np.linspace(1.1, 1.2, 200) + np.random.randn(200) * 0.0005
high = open_ + np.abs(np.random.rand(200) * 0.002)
low = open_ - np.abs(np.random.rand(200) * 0.002)
close = open_ + np.random.randn(200) * 0.0003
df = pd.DataFrame({'timestamp': rng, 'open': open_, 'high': high, 'low': low, 'close': close})
df.to_parquet('data/raw/sample_EURUSD_M1.parquet', index=False)
'@ | & $pythonExe -
    }
}

if (-not $InputParquet) {
    $InputParquet = Join-Path $projectRoot "data/raw/sample_EURUSD_M1.parquet"
}

$resolvedInput = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $InputParquet))
$resolvedOutputFeatures = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputFeatures))
$resolvedOutputLabeled = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputLabeled))

if (-not (Test-Path $resolvedInput)) {
    throw "Input Parquet file not found: $resolvedInput"
}

$featuresScript = Join-Path $projectRoot "scripts/feature_engineering_from_parquet.py"
Write-Host "[INFO] Running feature engineering: $resolvedInput -> $resolvedOutputFeatures"
& $pythonExe $featuresScript --input $resolvedInput --output $resolvedOutputFeatures

if (-not $SkipLabeling) {
    $labelsScript = Join-Path $projectRoot "scripts/generate_labels_from_features.py"
    Write-Host "[INFO] Running target labeling: $resolvedOutputFeatures -> $resolvedOutputLabeled"
    & $pythonExe $labelsScript --input $resolvedOutputFeatures --output $resolvedOutputLabeled --horizons "1,5,10,20" --threshold 0.0005 --sl 0.01 --tp 0.02
}

Write-Host "[INFO] Done. Output files:"
Write-Host "  - $resolvedOutputFeatures"
Write-Host "  - $resolvedOutputLabeled"
