param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ManifestPath = Join-Path $PSScriptRoot "MANIFEST_SHA256.txt"
if (Test-Path -LiteralPath $ManifestPath) {
    foreach ($Line in Get-Content -LiteralPath $ManifestPath) {
        if (-not $Line.Trim()) { continue }
        $Expected, $Relative = $Line -split "`t", 2
        if (-not $Relative) { throw "Invalid manifest line: $Line" }
        $Actual = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $Relative) -Algorithm SHA256).Hash
        if ($Actual -ne $Expected) { throw "SHA-256 mismatch: $Relative" }
    }
    Write-Host "SHA-256 manifest verified."
}
$PreviousPythonPath = $env:PYTHONPATH
$CodePath = (Resolve-Path "code").Path
$env:PYTHONPATH = if ($PreviousPythonPath) { "$CodePath;$PreviousPythonPath" } else { $CodePath }
try {
    & $Python -m compileall -q code
    if ($LASTEXITCODE -ne 0) { throw "Python compilation check failed." }
    & $Python -m unittest discover -s code -p "test_budget_policy.py"
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }
    & $Python code/audit_expanded_results.py
    if ($LASTEXITCODE -ne 0) { throw "Frozen-result integrity audit failed." }
    & $Python code/cluster_uncertainty.py
    if ($LASTEXITCODE -ne 0) { throw "Cluster-bootstrap reproduction failed." }
    Write-Host "Package verification passed."
} finally {
    if ($PreviousPythonPath) { $env:PYTHONPATH = $PreviousPythonPath }
    else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
}
