param(
    [string]$Python = "python",
    [switch]$Smoke,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($Smoke -eq $Full) { throw "Select exactly one of -Smoke or -Full." }

$Required = @(
    "data/raw/uci_steel/data.csv",
    "data/raw/uci_appliances/data.csv",
    "data/processed/uci_tetouan_hourly.csv",
    "data/processed/uci_ecl_hourly.csv",
    "data/processed/opsd_aggregate_hourly.csv"
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Path. Run download_and_prepare_data.ps1 first."
    }
}

$PreviousPythonPath = $env:PYTHONPATH
$CodePath = (Resolve-Path "code").Path
$env:PYTHONPATH = if ($PreviousPythonPath) { "$CodePath;$PreviousPythonPath" } else { $CodePath }
try {
    & $Python -m unittest discover -s code -p "test_budget_policy.py"
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }

    if ($Smoke) {
        & $Python code/rl_core_matrix.py --datasets steel --horizons 12 --budgets 108 --folds 1 --output results/smoke_core
        if ($LASTEXITCODE -ne 0) { throw "Smoke matrix failed." }
        & $Python code/rl_gate_ablation_matrix.py --datasets steel --horizons 12 --budgets 108 --folds 1 --output results/smoke_gate
        if ($LASTEXITCODE -ne 0) { throw "Smoke gate failed." }
        Write-Host "Smoke reproduction complete."
        exit 0
    }

    foreach ($Dataset in @("steel","tetouan","uci_ecl","opsd")) {
        & $Python code/rl_core_matrix.py --datasets $Dataset --horizons 6 12 24 48 --budgets 60 108 180 --folds 5 --output "results/rl_core_matrix_v2_$Dataset"
        if ($LASTEXITCODE -ne 0) { throw "Core matrix failed for $Dataset." }
        & $Python code/rl_gate_ablation_matrix.py --datasets $Dataset --horizons 6 12 24 48 --budgets 60 108 180 --folds 5 --output "results/rl_gate_ablation_v1_$Dataset"
        if ($LASTEXITCODE -ne 0) { throw "Gate matrix failed for $Dataset." }
    }
    & $Python code/rl_core_matrix.py --datasets appliances --horizons 6 12 24 48 --budgets 60 108 180 --folds 2 --output results/rl_core_matrix_v2_appliances
    & $Python code/rl_gate_ablation_matrix.py --datasets appliances --horizons 6 12 24 48 --budgets 60 108 180 --folds 2 --output results/rl_gate_ablation_v1_appliances
    & $Python code/rl_depth_experiments.py --datasets steel tetouan --horizons 12 --folds 5 --daily-budget 108 --skip-stagewise --skip-switching --output results/rl_depth_audit_v2
    & $Python code/rl_hyperparameter_grid.py --datasets steel tetouan --horizon 12 --folds 2 --daily-budget 108 --output results/rl_hyper_grid_v1
    foreach ($Penalty in @(0.01,0.05,0.10)) {
        $Suffix = if ($Penalty -eq 0.01) { "switch_0p01" } elseif ($Penalty -eq 0.10) { "switch_0p1" } else { "v1" }
        & $Python code/rl_sequential_environment.py --datasets steel tetouan --horizon 12 --folds 3 --daily-budget 108 --switching-penalty $Penalty --output "results/rl_sequential_audit_$Suffix"
        if ($LASTEXITCODE -ne 0) { throw "Sequential experiment failed at penalty $Penalty." }
    }
    & $Python code/summarize_expanded_results.py
    & $Python code/cluster_uncertainty.py
    & $Python code/audit_expanded_results.py
    & $Python code/plot_expanded_rl_figures.py
    & $Python code/generate_expanded_tables.py
    Write-Host "Full reproduction complete."
} finally {
    if ($PreviousPythonPath) { $env:PYTHONPATH = $PreviousPythonPath }
    else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
}
