# Student Name: YOUR_NAME_HERE
# Index Number: YOUR_INDEX_NUMBER_HERE
#
# Double-click or run in PowerShell from anywhere:
#   powershell -ExecutionPolicy Bypass -File "C:\path\to\run_web.ps1"
#
# Starts the Streamlit web UI in your default browser.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Missing .venv. From this folder run:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

if (-not $env:OPENAI_API_KEY) {
    Write-Host "Note: OPENAI_API_KEY is not set. Chat answers need an API key."
    Write-Host 'Example: $env:OPENAI_API_KEY = "sk-..."'
}

Write-Host "Starting Streamlit (your browser should open to the app URL)..."
& $py -m streamlit run app.py
