$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourcePath = Join-Path $ProjectRoot "src"
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$SourcePath;$env:PYTHONPATH" } else { $SourcePath }

python -m hydroswarm.cli self-test
exit $LASTEXITCODE

