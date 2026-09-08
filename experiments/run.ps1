param([ValidateSet('metrics', 'model', 'decode', 'tests')][string]$Mode = 'metrics')
$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path $PSScriptRoot -Parent
$experimentPython = Join-Path (Split-Path $experimentRoot -Parent) 'Insideai/backend/.venv/Scripts/python.exe'
$previousPythonPath = $env:PYTHONPATH
Push-Location $experimentRoot
try {
    $env:PYTHONPATH = "$experimentRoot;$experimentRoot/.experiment-deps"
    if ($Mode -eq 'metrics') { & $experimentPython experiments/metric_controls.py }
    elseif ($Mode -eq 'model') { & $experimentPython experiments/layer_probe.py }
    elseif ($Mode -eq 'decode') { & $experimentPython experiments/decode_probe.py }
    else { & $experimentPython -m pytest tests/test_chronos_lock.py tests/test_bocpd.py tests/test_dual_link.py tests/test_spectral_gap_reference.py -q --tb=short }
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
