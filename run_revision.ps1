param([string]$Python = "python", [switch]$Rerun)
$ErrorActionPreference = "Stop"
$PythonExe = (Get-Command $Python -ErrorAction Stop).Source
Push-Location (Join-Path $PSScriptRoot "original")
try {
    $Scripts = if ($Rerun) { @("run_revision.py", "summarize_revision.py", "diagnose_uci.py", "cost_sensitivity.py", "audit_revision.py") } else { @("audit_revision.py") }
    foreach ($Script in $Scripts) {
        & $PythonExe (Join-Path $PSScriptRoot "revision_experiments/$Script")
        if ($LASTEXITCODE -ne 0) { throw "Failed: $Script" }
    }
} finally { Pop-Location }
