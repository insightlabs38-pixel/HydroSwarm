param(
    [string]$Address = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourcePath = Join-Path $ProjectRoot "src"
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$SourcePath;$env:PYTHONPATH" } else { $SourcePath }

python -m hydroswarm.cli start --host $Address --port $Port
exit $LASTEXITCODE

