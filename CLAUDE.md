# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AFO (Australian Farm Optimisation Model) is a whole-farm bioeconomic linear-programming (LP) model
built with Pyomo. It represents an Australian dryland farming system (rotations, crops, pastures,
livestock, stubble, supplementary feed, machinery, labour, finance) and solves for the profit-maximising
combination of activities. This is a research/economics model, not an application with a test suite —
correctness is judged by whether the LP solves to the expected optimum and whether trial outputs match
prior known-good runs, not by unit tests.

## Running the model

There is no build step; this is a plain Python/Pyomo project run directly with a Python interpreter that
has `requirements.txt` installed (a `venv` directory already exists in the repo root — activate it, or use
your own).

- **Run trials**: `python RunAfoRaw_Multi.py <experiment_group> <processors>` — multiprocess entry point.
  Optional args: which `Exp Group` number from `ExcelInputs/exp.xlsx` to run, and how many processes.
  With no args, experiment group `0` runs on 1 processor. `RunAfoRaw.py` is the single-process
  equivalent (used for debugging since it's easier to step through).
- **Run reports**: `python RunReportsRaw.py <experiment_group> <processors> <report_group>` — a trial
  must have already been run (its outputs pickled to `pkl/`) before it can be reported on.
  `ReportControlsUser.py` (gitignored, user-local) can customise report settings.
  `ReportControlsUser-Google.py` is a committed template for the Google/cloud variant.
- **Solver**: set via `solver_method` in the run script — options are `CPLEX`, `glpk`, `cbc`, `ipopt`,
  `HiGHS` (default/open-source, via `highspy`). CPLEX is proprietary but has advanced features used
  historically; GLPK/HiGHS give equivalent answers.
- **Experiment control**: `ExcelInputs/exp.xlsx` drives everything — which trials run (`Run Trials`
  column), which experiment group they belong to (`Exp Group` column), and the sensitivity analysis (SA)
  values applied per trial. Trials are only re-run if `exp.xlsx`, an input file, or relevant code changed
  since the last run (`force_run = True` in the run script overrides this and forces all trials to run).
- **Debugging a single module in isolation**: `lib/AfoLogic/TestMaster.py` loads one trial's inputs and
  lets you selectively call individual precalc modules (`test_pas`, `test_sup`, `test_stock`, etc.) without
  running the full model — useful for isolating a slow or broken module.
- **Docs**: Sphinx docs live in `docs/`; build with `docs/Makefile` (`make html` from `docs/`). Hosted on
  Read the Docs per `.readthedocs.yaml`.
- **Calibrate sheep-generator coefficients**: `python -u lib/AfoLogic/SheepCalibration.py <exp_no>
  <n_processes> <calibrate|report>` — a separate entry point (not part of the `AfoInit.exp()` pipeline
  above) that fits `StockGenerator` coefficients to observed data using scipy's `differential_evolution`.
  Targets, weights, best-bet starting values and bounds come from `ExcelInputs/Calibration_control.xlsx`
  (sheets `Targets`/`Weights`/`BestBet`/`Low`/`High`); coefficients being fit normally live in
  `Universal.xlsx` but are *not* written back there automatically — results (fitted coefficients, success,
  weighted SSE, iterations) are written to `ExcelInputs/CalibrationResults.xlsx` for manual review/transfer.
  Coefficient sets are grouped into independent "teams" (e.g. different traits/genotypes) that can each run
  DE in parallel; `calibration_objective()` calls `StockGenerator.generator(coefficients_c=..., calibrate_trait_values=True, ...)`
  directly as the DE objective function (skipping the Pyomo/LP stages entirely) and returns the
  weighted sum-of-squared-error between simulated and target trait values. Because `StockGenerator` reads
  and mutates shared module state (`sinp`/`uinp`/`pinp`), the trial inputs are reset before every DE
  objective evaluation (see `calibration_objective`/`setup_trial`).

There is no linter or automated test suite configured in this repo (no pytest/tox/pyproject.toml). Validate
changes by running a trial and comparing the variable summary / profit against a known-good baseline, or by
diffing the generated `.lp` file.

## Architecture

### Execution pipeline (per trial, see `lib/AfoLogic/AfoInit.py::exp()`)

1. **Inputs** — three Excel workbooks in `ExcelInputs/`: property inputs (region/farm-specific, e.g.
   `Property_CWW.xlsx`), universal inputs (global prices etc., `Universal.xlsx`), and structural inputs
   (core model structure, `Structural.xlsx`). Loaded via `lib/RawVersion/LoadExcelInputs.py` into
   `sinp`/`uinp`/`pinp` modules (`StructuralInputs.py`, `UniversalInputs.py`, `PropertyInputs.py`), which
   act as process-wide singletons reset at the start of every trial (`f_select_n_reset_*`).
2. **Sensitivity analysis (SA)** — `lib/AfoLogic/Sensitivity.py` holds six dicts (`sam`, `sap`, `saa`,
   `sat`, `sav`, `sar` — multiply/percent/add/... adjustments) that `exp.xlsx` populates per trial via
   `RawVersionExtras.f_process_user_sa`. Any module reading an input value that might be swept in
   experiments applies the relevant `sa*` dict through `Functions.f_sa`.
3. **Precalcs** — one module per farm-system component converts raw inputs + SA into the numeric arrays
   Pyomo needs (e.g. `Phase.py`/`PhasePyomo.py`, `Mach.py`/`MachPyomo.py`, `StockGenerator.py`/
   `StockPyomo.py`, `Pasture.py`/`PasturePyomo.py`). The convention throughout the codebase is a
   `<Module>.py` (biology/economics calculations, produces numpy arrays) paired with a
   `<Module>Pyomo.py` (wraps those arrays into Pyomo `Param`/`Set`/`Var`/`Constraint` objects). Precalc
   order matters and is enforced in `AfoInit.py` — e.g. `StockPyomo` must run before `SupFeedPyomo`,
   `CropGrazingPyomo`, `SaltbushPyomo`, `CropResiduePyomo`, `PasturePyomo` because they consume the `nv`
   (nutritive value) dict it populates.
   Arrays use short letter-code axes (season, land use, LMU, generator period, nutrition profile, etc.) —
   see `docs/axes.rst` for the full key. Multi-letter axis suffixes on variable names (e.g. `v_phase_area_l`,
   arrays named `*_p6z` etc.) tell you which axes that object is indexed over.
4. **Pyomo model build** — a fresh `pe.ConcreteModel()` is built every trial (module `CreateModel.py` for
   sets, then each `<Module>Pyomo.py::f1_*pyomo_local` in a fixed order matching precalc dependencies,
   `BoundsPyomo.py` last since it references sets built by everything else).
5. **Solve** — `CorePyomo.py::coremodel_all` assembles the objective and remaining constraints and calls
   the solver.
6. **Outputs** — `lib/RawVersion/SaveOutputs.py` pickles `lp_vars`/`r_vals` per trial to `pkl/`; reporting
   (`RunReportsRaw.py` → `lib/AfoLogic/ReportControl.py` / `ReportFunctions.py`) reads those pickles and
   writes Excel report summaries.

### Directory layout

- `lib/AfoLogic/` — the model itself: precalc modules, Pyomo modules, sets, reporting, sheep generator
  (`StockGenerator.py` is the largest/most complex module — biology of livestock at a fine time step).
  `PastureNew/` holds an in-progress alternative pasture representation not yet wired into the main flow.
- `lib/RawVersion/` — the "raw" (as opposed to the hosted web-app) entry-point glue: loading `exp.xlsx`
  and the Excel input workbooks, applying SA from the experiment sheet, saving outputs.
- `Inputs/` — Python-level input modules for the tree/agroforestry component (`TreePropertyInputs.py`,
  `TreeUniversalInputs.py`), separate from the Excel-driven inputs.
- `ExcelInputs/` — all Excel workbooks (property/universal/structural inputs, `exp.xlsx`, calibration and
  simulation workbooks). Many files here are ad hoc backups/scratch copies kept by convention, not all are
  live inputs — `Property_*.xlsx`, `Universal.xlsx`, `Structural.xlsx`, and `exp.xlsx` are the canonical
  ones referenced by the run scripts.
- `docs/` — Sphinx documentation, including the mathematical/model logic documentation
  (`docs/model_doc.rst` is the toctree root) which is the authoritative description of the LP formulation
  per subsystem (rotation, phase/cropping, feedsupply, finance, labour, livestock, machinery, emissions,
  seasonality). Read the relevant `docs/<Module>.rst` / `docs/<Module>Pyomo.rst` pair before modifying a
  subsystem — they document the intended math, which the code should match.
- `pkl/`, `Output/`, `log/` — generated/gitignored run artifacts (pickled trial results, `.lp` files,
  variable summaries). Not source.

### Key conventions

- Function naming: `f_` prefix for most functions, `f1_` for functions specifically building/wiring Pyomo
  components (constraint/param builders), matching the module-pair convention above.
- Season/sequence axes (`q`, `s`, `z`, `p7`) implement AFO's representation of seasonal risk — a trial can
  represent multiple discrete season types and/or a sequence of years; see `docs/seasonality_index.rst`
  and `docs/ShortTermDecisions_index.rst` before touching anything with a `z` or `q` axis.
- Global module-level state (`sinp`, `uinp`, `pinp`, `sen`) is deliberately mutated and reset once per
  trial rather than passed as objects — this is a long-standing pattern in the codebase, not a bug; when
  reading an array from these modules be aware it reflects only the *current* trial.
- See `docs/debugging.rst` for a catalogue of known failure modes (infeasibility from tiny/NaN coefficients,
  stale constraints referencing rebuilt variables, mismatched weight distributions causing livestock to
  drop out of the solution, etc.) before assuming a new bug is novel.
