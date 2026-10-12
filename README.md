# Code and Experimental Artifacts

**Manuscript:** *Compute-Constrained Adaptive Electricity Load Forecasting via Offline Policy Learning*  
**Authors:** Jiapei Liu, Zhigang Zhao, Li Xu, and Zhaoyuan Zhu  
**Version:** 1.0.0 (10 October 2026)

This repository contains reproducibility code and frozen outputs for the formal study, together with post-review supplementary analyses used in the revised manuscript. The original formal result snapshot is retained under `original/`; reviewer-requested experiments and their protocols are under `revision_experiments/`.

## Contents

- `original/code/`: forecasting experts, policy implementations, preprocessing, audits, and figure/table generation.
- `original/results/`: frozen outputs for the 960 formal policy evaluations and associated analyses.
- `original/data/`: source instructions and retained profiles. Raw source datasets are not redistributed; follow [`original/data/DATA_SOURCES.md`](original/data/DATA_SOURCES.md) and source terms.
- `original/artifacts/`: regenerated figures and tables.
- `revision_experiments/`: reviewer-requested MLP, UCI–ECL diagnostic, non-RL baseline, and cost-vector sensitivity materials, with scripts, protocols, and archived outputs.
- `run_revision.ps1`, `original/run_reproduction.ps1`, `original/verify_package.ps1`: launchers and verification.
- `MANIFEST_SHA256.txt`: integrity manifest for the tracked release files.

## Reproduction

The audited environment for the original package is Windows 11 and Python 3.12; dependency pins are provided in `requirements.txt` and `original/requirements.txt`. Run `original/verify_package.ps1` for the frozen-result integrity audit. Dataset downloads and preprocessing instructions are in [`original/data/DATA_SOURCES.md`](original/data/DATA_SOURCES.md). The full model-fitting matrix is compute-intensive; scripts support resumable execution. Revision scripts and design limitations are documented in `revision_experiments/PROTOCOL.md` and `revision_experiments/COST_SENSITIVITY_ADDENDUM.md`.

### Quick start (PowerShell)

```powershell
python -m pip install -r requirements.txt
./run_revision.ps1
```

The default revision command audits archived outputs. To regenerate them, use `./run_revision.ps1 -Rerun`; first obtain the third-party datasets with `original/download_and_prepare_data.ps1` and consult the dataset source notes. For the formal study, use `original/run_reproduction.ps1 -Smoke` or `-Full` after preparing data.

The cost values are relative experimental credits, not platform-independent latency, energy, or monetary measurements. The alternative base-cost refit is a focused sensitivity analysis and does not establish robustness across hardware or arbitrary cost profiles.

The archived non-RL audit covers Steel, Tetouan, and UCI–ECL (five folds each) plus two descriptive Appliances folds at 12 h and budget 108; it does not contain OPSD. The hour schedule uses calendar information and respects the action budget. The disagreement rule computes its trigger from all four candidate forecasts but counts only the selected action's cost; its reported gain is a retrospective diagnostic, not a deployment-feasible result under the same full call accounting.

## Rights and citation

Software source code (`.py` files in `original/code/` and `revision_experiments/`, plus `.ps1` launchers under `original/` and `revision_experiments/`) is licensed under MIT; see `LICENSE-MIT.txt`. Documentation, profiles, frozen result tables, and derived experimental artifacts are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSES.md` for scope and third-party data notices. Raw source datasets are not included.

Cite the manuscript and software using `CITATION.cff`. The version 1.0.0 code and derived artifacts are permanently archived at https://doi.org/10.5281/zenodo.23231947.
