# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John

SheepCalibration.py is run using python -u SheepCalibration.py {<exp no>} {<number of processes>} {<calibrate/report>}
<exp no> is an optional argument, if it is excluded the default trial is QT
<number of processes> is an optional argument. The default is 1 (don't multiprocess teams use MAX_WORKERS_WITHIN_TEAM on each team)
<calibrate/report> is optional, default True. True is calibrate, False is report the trait values.
<method> is optional: de, lsq or lsq+de (see METHOD). If blank, sav['calib_method'] in exp.xlsx or METHOD is used.

Multiprocessing the teams (with workers =1) will be quicker if
 (1) the number of cpus is greater than population size - Population size is pop_size parameter (5 to 15) x number of coefficients
 (2) there are sufficient teams to occupy the computer resource
If not, use multiple workers. The maximum useful number of workers is the size of the selection population
Multiprocessing teams can be more efficient because it can use 'immediate' updating

sys.argv: Experiment number (will use the first trial in the experiment). If blank uses QT (trial 12)
          Number of multi processes. If blank will not multiprocess but will use workers
          Calibrate/report. If blank will calibrate
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from timeit import default_timer as timer
import sys
import os

##Calibration specific imports
import pandas as pd
from scipy import optimize as spo
import multiprocessing as mp
import time

#sets the path to the root directory so the relative imports in the other files work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lib.RawVersion import LoadExcelInputs as dxl
from lib.RawVersion import LoadExp as exp
from lib.RawVersion import RawVersionExtras as rve
from lib.AfoLogic import StructuralInputs as sinp
from lib.AfoLogic import PropertyInputs as pinp
from lib.AfoLogic import UniversalInputs as uinp
from lib.AfoLogic import Functions as fun
from lib.AfoLogic import Sensitivity as sen
from lib.AfoLogic import relativeFile
from lib.AfoLogic import StockGenerator as sgen

###############
#User control #
###############
# Command line: python SheepCalibration.py <exp_no> <n_processes> <RUN_CALIBRATION>
# RUN_CALIBRATION: True is to run the calibration, False is report the trait values
try:
    RUN_CALIBRATION = sys.argv[3].lower() in ['true', '1', 'yes']
except (IndexError, AttributeError):
    RUN_CALIBRATION = True  # default

##configure logging as early as possible (before StockGenerator's logger.debug() calls can fire) so
## calibration runs suppress StockGenerator debug output while report runs show it.
## stream=sys.stdout so log lines share the print() stream and stay in order (stderr and stdout are
## buffered separately so the obj print could otherwise appear in the middle of the trait output).
## root is WARNING so third-party libraries (pyomo/matplotlib) stay quiet; AFO's own loggers (under 'lib') are DEBUG for report runs.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format='%(processName)s %(name)s %(levelname)s: %(message)s' if RUN_CALIBRATION else '%(message)s'
)
logging.getLogger('lib').setLevel(logging.WARNING if RUN_CALIBRATION else logging.DEBUG)

# Optimisation controls. Team multiprocessing always uses one worker per team;
# single-team mode can use workers within scipy's differential_evolution.
MAXITER = 400
POPSIZE = 5   # between 5 and 15. population = popsize * n_coeff
MAX_WORKERS_WITHIN_TEAM = 10   #48  #setting workers to a factor of 'population' ensures even utilisation of CPUs
                                 #only takes effect when team_processes is False (n_processes==1) - team
                                 #multiprocessing always forces workers=1 per team regardless of this value
# POLISH, TOL & ATOL are defaults that can be overridden per trial in exp.xlsx using sav['calib_polish'],
# sav['calib_tol'] & sav['calib_atol'] (see build_context).
# scipy's polish step (L-BFGS-B) is single-threaded so it can't use the DE worker pool, and is a local
# (finite-difference) refinement that assumes a smooth objective.
# If polish is used DE can stop looser (e.g. tol=0.05, atol=0.05) because polish finishes the job. If polish
# is not used DE has to do all the work itself so it needs tighter values (e.g. tol=0.001, atol=0.01) to
# reach an equivalent final precision.
POLISH = True
TOL = 0.05
ATOL = 0.01   #If the trait weights are 100 * estimated EV then SQRT(Obj) is the error in the EV of the target trait
# The initial population is split between members near BestBet and members spread over the full bounds (see
# f_initial_population). With best1bin each trial is best + F*(r1-r2), so if the whole population is spread over wide
# bounds the difference vectors are large and nearly every trial lands far from BestBet. Starting part of the population
# near BestBet makes the early difference vectors small, while the spread members keep the full range of the bounds
# available (e.g. for CFW coefficients that need to move a long way when LW is also being calibrated).
INIT_NEAR_PROPN = 0.5    #proportion of the population sampled near BestBet (0 = Latin hypercube over the full bounds only)
INIT_NEAR_SD = 0.05      #standard deviation of the near members as a proportion of each coefficient's range
# Calibration method, overridden by sav['calib_method'] in exp.xlsx or the 4th command line argument.
# 'de' is differential evolution: a global search, thousands of generator evaluations per team.
# 'lsq' is bounded Gauss-Newton least squares from BestBet: a local search, tens to a few hundred evaluations per team.
#   The weighted trait errors are linearised with the Jacobian (traits w.r.t. coefficients) and each step solves that
#   linear problem within the bounds and a trust region. The Jacobian starts as the one at the Standard (measured once
#   and shared by the teams), is updated from each step (Broyden), and is re-measured at the team's point (one evaluation
#   per coefficient) when a step fails to halve the objective, so curved traits (e.g. fat) still converge. It needs a
#   reasonable BestBet (e.g. the BBMatrix of the calibration Master, v12w on). A target that can't be reached ends at the
#   best compromise within the bounds.
#   Bounds: with a cLimits sheet in Calibration_control.xlsx (Master v12z on: the biological limits of each coefficient
#   adjustment, blank = no limit) lsq is bounded by those limits and Low/High only set its scale (the Jacobian step and the
#   trust region, in half-ranges of Low/High). Without the sheet Low/High are both the bounds and the scale.
# 'lsq+de' runs lsq, then DE from the lsq result for the teams still above LSQ_DE_ABOVE.
#   DE searches Low/High (clipped to the cLimits). If lsq ended outside Low/High, DE's range is re-centred on the lsq result.
METHOD = 'lsq'
LSQ_STEP = 0.05        #finite-difference step for the Jacobian, as a proportion of each coefficient's Low/High half-range (doubled for the central differences at the Standard)
LSQ_TARGET = 1e-5      #stop when the objective is below this
LSQ_MAX_EVAL = 600     #maximum generator evaluations per team
LSQ_DE_ABOVE = 0.005   #'lsq+de': teams with an objective above this after lsq continue with DE

worker_context = {}


def get_sa_value(user_sa, key, default=False):
    result = default
    for item in user_sa:
        if item["key1"] == key and item["value"] != '-':
            result = item["value"]
    return result


def get_trial_sa(exp_data, trial):
    return rve.f_process_user_sa(exp_data, trial)


def load_trial_fs(user_sa):
    """Load the feed-supply pickle requested by the trial sensitivity settings."""
    fs_use_pkl = get_sa_value(user_sa, "fs_use_pkl", False)
    fs_use_number = get_sa_value(user_sa, "fs_use_number", None)
    return dxl.f_load_fs(fs_use_pkl, fs_use_number)


def reset_trial_inputs(trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults, user_sa=None):
    """Reset all mutable AFO input modules back to the selected trial state."""
    ##select property for the current trial
    property = trial_pinp.iloc[trial]

    ##process user SA
    if user_sa is None:
        user_sa = get_trial_sa(exp_data, trial)

    ##select property and reset default inputs for the current trial. Must occur first.
    sinp.f_select_n_reset_sinp(sinp_defaults)
    sinp.f_landuse_sets()
    uinp.f_select_n_reset_uinp(uinp_defaults)
    pinp.f_select_n_reset_pinp(property, pinp_defaults)

    ##update sensitivity values
    sen.create_sa()
    fun.f_update_sen(user_sa, sen.sam, sen.saa, sen.sap, sen.sar, sen.sat, sen.sav)

    ##call sa functions - assigns sa variables to relevant inputs
    sinp.f_structural_inp_sa(sinp_defaults)
    uinp.f_universal_inp_sa(uinp_defaults)
    pinp.f_property_inp_sa(pinp_defaults)

    ##expand p6 axis to include nodes
    pinp.f1_expand_p6()

    ##mask lmu
    pinp.f1_mask_lmu()
    return user_sa


def setup_trial(trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults):
    """Reset trial inputs and load the feed-supply data for one generator run."""
    user_sa = reset_trial_inputs(trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults)
    pkl_fs = load_trial_fs(user_sa)
    return pkl_fs


def read_calibration_control():
    df_targets_tp = pd.read_excel(relativeFile.findExcel("Calibration_control.xlsx"), sheet_name="Targets", index_col=[0], header=[0], engine='openpyxl')
    df_weights_p = pd.read_excel(relativeFile.findExcel("Calibration_control.xlsx"), sheet_name="Weights", index_col=[0], header=[0], engine='openpyxl')
    df_bestbet_tc = pd.read_excel(relativeFile.findExcel("Calibration_control.xlsx"), sheet_name="BestBet", index_col=[0], header=[0], engine='openpyxl')
    df_bnd_lo_tc = pd.read_excel(relativeFile.findExcel("Calibration_control.xlsx"), sheet_name="Low", index_col=[0], header=[0], engine='openpyxl')
    df_bnd_up_tc = pd.read_excel(relativeFile.findExcel("Calibration_control.xlsx"), sheet_name="High", index_col=[0], header=[0], engine='openpyxl')
    ##biological limits of the coefficient adjustments (rows Lower & Upper, blank = no limit). Optional: control files made
    ## before Master v12z don't have the sheet and Low/High are then the only bounds.
    try:
        df_lim_c = pd.read_excel(relativeFile.findExcel("Calibration_control.xlsx"), sheet_name="cLimits", index_col=[0], header=[0], engine='openpyxl')
    except ValueError:
        df_lim_c = None
    return df_targets_tp, df_weights_p, df_bestbet_tc, df_bnd_lo_tc, df_bnd_up_tc, df_lim_c


def check_calibration_control(df_targets_tp, df_weights_p, df_bestbet_tc, df_bnd_lo_tc, df_bnd_up_tc, df_lim_c=None):
    """Check the Calibration_control.xlsx sheets line up before starting the calibration.

    The coefficients and traits are passed to the generator by position (not by name), so a column that is
    missing or out of order in one sheet shifts every following value. Rows are compared by count because
    the row labels are not the same in every sheet.
    """
    problems = []
    ##coefficient sheets must have identical columns (BestBet is the x0 for DE, Low/High are the bounds)
    for name, df in [("Low", df_bnd_lo_tc), ("High", df_bnd_up_tc)] + ([("cLimits", df_lim_c)] if df_lim_c is not None else []):
        if list(df.columns) != list(df_bestbet_tc.columns):
            only_bestbet = [c for c in df_bestbet_tc.columns if c not in df.columns]
            only_other = [c for c in df.columns if c not in df_bestbet_tc.columns]
            problems.append(f"{name} columns differ from BestBet ({df.shape[1]} vs {df_bestbet_tc.shape[1]})."
                            f" Only in BestBet: {only_bestbet}. Only in {name}: {only_other}."
                            + (" Same names but a different order." if not only_bestbet and not only_other else ""))
    ##one row per team in Targets and each coefficient sheet
    n_teams = df_targets_tp.shape[0]
    for name, df in [("BestBet", df_bestbet_tc), ("Low", df_bnd_lo_tc), ("High", df_bnd_up_tc)]:
        if df.shape[0] != n_teams:
            problems.append(f"{name} has {df.shape[0]} team rows but Targets has {n_teams}.")
    ##one weight for each target trait
    if df_weights_p.shape[1] != df_targets_tp.shape[1]:
        problems.append(f"Weights has {df_weights_p.shape[1]} trait columns but Targets has {df_targets_tp.shape[1]}.")
    else:
        ##same trait in each column (not case sensitive). The generator also checks the Targets names against its own list.
        for c, (name_w, name_t) in enumerate(zip(df_weights_p.columns, df_targets_tp.columns)):
            if str(name_w).strip().lower() != str(name_t).strip().lower():
                problems.append(f"Weights column {c} is '{name_w}' but Targets column {c} is '{name_t}'.")
    ##bounds must be valid and contain the best bet (only checked if the shapes line up)
    if not problems:
        lo, up, bestbet = df_bnd_lo_tc.values, df_bnd_up_tc.values, df_bestbet_tc.values
        for desc, mask in [("Low > High", lo > up), ("BestBet outside Low/High", (bestbet < lo) | (bestbet > up))]:
            for t, c in zip(*np.nonzero(mask)):
                problems.append(f"{desc}: team row {t} ({df_targets_tp.index[t]}), coefficient {df_bestbet_tc.columns[c]}.")
    ##the biological limits must be valid (Low/High may go past them: DE's range is clipped to them). A BestBet outside
    ## them is only a warning: the calibration starts from the BestBet clipped to the limits.
    if not problems and df_lim_c is not None:
        if list(df_lim_c.index[:2].str.lower()) != ['lower', 'upper']:
            problems.append(f"cLimits rows must be Lower and Upper, not {list(df_lim_c.index)}.")
        else:
            lim_lo, lim_up = f1_limits(df_lim_c)
            for c in np.flatnonzero(lim_lo > lim_up):
                problems.append(f"cLimits Lower > Upper: coefficient {df_bestbet_tc.columns[c]}.")
            for t, c in zip(*np.nonzero((bestbet < lim_lo) | (bestbet > lim_up))):
                print(f"Warning: BestBet outside cLimits, team row {t} ({df_targets_tp.index[t]}), coefficient"
                      f" {df_bestbet_tc.columns[c]}: {bestbet[t, c]:g} is clipped to [{lim_lo[c]:g}, {lim_up[c]:g}].")
    if problems:
        raise ValueError("Calibration_control.xlsx sheets do not line up:\n  " + "\n  ".join(problems))


def f1_limits(df_lim_c):
    """Lower and Upper rows of cLimits as arrays, a blank (no limit) as -inf/+inf."""
    lim = df_lim_c.apply(pd.to_numeric, errors='coerce').values.astype(float)
    return np.nan_to_num(lim[0], nan=-np.inf), np.nan_to_num(lim[1], nan=np.inf)


def init_worker(context):
    """Store per-trial data in each team worker.

    The actual trial reset happens inside calibration_objective() before every
    sgen.generator call. This initializer only gives the worker enough context
    to run multiple teams without re-pickling that context for every task.
    """
    context = context.copy()
    worker_context.clear()
    worker_context.update(context)
    ##on Windows the workers are spawned without python's -u, so their output would sit in a buffer until it fills or
    ## the worker ends. Line buffering writes each print as it is made (in one write, so lines from teams don't mix).
    sys.stdout.reconfigure(line_buffering=True)


def f_initial_population(bestbet, bounds, population, seed=None):
    """Initial DE population: INIT_NEAR_PROPN of the members near BestBet, the rest a Latin hypercube over the bounds.

    Passed to differential_evolution as init, so popsize is not used (the population is the number of rows).
    x0 (BestBet) still replaces the first member.
    """
    rng = np.random.default_rng(seed)
    lo, up = np.array(bounds, dtype=float).T
    n_coef = lo.size
    n_near = int(round(population * INIT_NEAR_PROPN))
    n_wide = population - n_near
    ##near members: normal around BestBet with sd a proportion of each coefficient's range, clipped to the bounds
    near = np.clip(bestbet + rng.normal(0, INIT_NEAR_SD * (up - lo), (n_near, n_coef)), lo, up)
    ##wide members: Latin hypercube - one member in each of n_wide equal strata of each coefficient, strata shuffled independently
    strata = np.argsort(rng.random((n_wide, n_coef)), axis=0)
    wide = lo + (strata + rng.random((n_wide, n_coef))) / n_wide * (up - lo)
    return np.vstack([near, wide])


def f1_team_context(t, context):
    """Context for evaluating team t: the team's targets and weights, and the trial's SA and feed supply."""
    team_context = context.copy()
    team_context["team"] = t
    team_context["calibration_weights"] = context["weights_p"]
    team_context["calibration_targets"] = context["targets_tp"][t]
    # pkl_fs only depends on user_sa (fixed for the whole team), not on the coefficients
    # being tested, so it's safe to load it once here rather than on every evaluation.
    user_sa = get_trial_sa(context["exp_data"], context["trial"])
    team_context["user_sa"] = user_sa
    team_context["pkl_fs"] = load_trial_fs(user_sa)
    return team_context


def run_calibration_for_team(t, context):
    """Calibrate one team with the selected method (see METHOD).

    This is used by both run modes:
    - sequential loop over teams and workers >= 1
    - multiprocessing loop where each process handles a team and workers = 1
    """
    lo, up = context["bnd_lo_tc"][t], context["bnd_up_tc"][t]
    ##biological limits (cLimits), else Low/High. lsq is bounded by the limits and scaled by the Low/High half-range;
    ## DE searches Low/High clipped to the limits.
    lim_lo, lim_up = (context["lim_lo_c"], context["lim_up_c"]) if context.get("lim_lo_c") is not None else (lo, up)
    half = (up - lo) / 2
    bestbet = np.clip(context["bestbet_tc"][t], lim_lo, lim_up)
    team_context = f1_team_context(t, context)
    team_context["progress"] = {"step": 0, "best": np.inf}
    print(f"Starting calibration for Team {t}...")
    if context["method"] == 'de':
        ##(widened to the clipped BestBet if Low/High lie wholly beyond a limit)
        de_lo = np.minimum(np.maximum(lo, lim_lo), bestbet)
        de_up = np.maximum(np.minimum(up, lim_up), bestbet)
        result = f1_run_de(t, team_context, bestbet, de_lo, de_up)
    else:
        result = f1_run_lsq(t, team_context, bestbet, lim_lo, lim_up, half)
        if context["method"] == 'lsq+de' and result[3] > LSQ_DE_ABOVE:
            print(f"Team {t} objective {result[3]:g} after lsq is above {LSQ_DE_ABOVE}: continuing with DE from the lsq result")
            ##DE's range is Low/High, re-centred on the lsq result for a coefficient that lsq took outside it
            x = result[1]
            outside = (x < lo) | (x > up)
            de_lo = np.maximum(np.where(outside, x - half, lo), lim_lo)
            de_up = np.minimum(np.where(outside, x + half, up), lim_up)
            result_de = f1_run_de(t, team_context, x, de_lo, de_up)
            result = result_de if result_de[3] < result[3] else result
    ##all the trait values at the final coefficients (for the Traits sheet of CalibrationResults.xlsx)
    return result + (f1_final_traits(result[1], team_context),)


def f1_final_traits(x, team_context):
    """All the trait values at a team's final coefficients, evaluated as in report mode (offspring included) so the
    offspring traits are real too. One extra (slower) evaluation per team; it replaces a separate report-mode run."""
    reset_trial_inputs(team_context["trial"], team_context["exp_data"], team_context["trial_pinp"]
                       , team_context["sinp_defaults"], team_context["uinp_defaults"], team_context["pinp_defaults"]
                       , user_sa=team_context["user_sa"])
    out = {}
    try:
        sgen.generator(coefficients_c=x, params={}, r_vals={}, nv={}, pkl_fs_info={}, pkl_fs=team_context["pkl_fs"]
                       , o_trait_values=out, stubble=team_context["stubble"]
                       , calibration_weights_p=team_context["calibration_weights"]
                       , calibration_targets_p=team_context["calibration_targets"]
                       , calibration_trait_names_p=team_context["trait_names_p"])
    except Exception as e:
        print(f"Warning: the trait values of Team {team_context.get('team')} could not be evaluated. {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return None
    return out.get("output")


def f1_run_de(t, team_context, x0, lo, up):
    """Differential Evolution for one team, starting from x0 (BestBet, or the lsq result for 'lsq+de')."""
    bounds = list(zip(lo, up))
    bestbet = x0
    context = team_context

    ##report each DE step with the team number. DE's own disp output doesn't include the team, so when the teams run
    ## in parallel their steps can't be told apart. The callback (scipy 1.11) isn't passed the objective, so the best
    ## objective is tracked in calibration_objective. This only works if the objective runs in this process
    ## (workers = 1, as when multiprocessing the teams), otherwise f(x) is reported as n/a.
    ##convergence is tol / (relative spread of the population objective). DE stops when it reaches 1 (or earlier if atol is met).
    ##the best coefficients (xk) are also printed on one line at full precision, so if a run is stopped early (or a rerun
    ## is started from where a team got to) the latest best can be recovered from the log. Each line is written in one
    ## call (newline included) so lines from teams running in parallel don't get mixed together (python -u writes the
    ## text and print's end separately).
    ##the step count continues from an lsq run ('lsq+de') so the latest step in the log is the latest point
    def f1_report_step(xk, convergence):
        progress = team_context["progress"]
        progress["step"] += 1
        best = f"{progress['best']:g}" if context["workers"] == 1 else "n/a"
        print(f"Team {t} step {progress['step']}: f(x)= {best} convergence= {convergence:.3g}\n"
              f"Team {t} step {progress['step']} x: {xk.tolist()}\n", end='')

    population = context["popsize"] * len(bestbet)
    init = f_initial_population(bestbet, bounds, population) if INIT_NEAR_PROPN > 0 else 'latinhypercube'

    result = spo.differential_evolution(
        calibration_objective,
        bounds,
        args = (team_context,),
        maxiter=context["maxiter"],
        popsize=context["popsize"],
        tol=context["tol"],
        atol=context["atol"],
        disp=context["disp"],
        polish=context["polish"],
        updating=context["updating"],
        workers=context["workers"],
        x0=bestbet,
        init=init,
        callback=f1_report_step,
        #best1bin mutates every individual relative to the current best (best + F*(r1-r2)), the greedy
        #default strategy suited to this non-multimodal problem.
        #Also tested currenttobest1bin but it was slower to converge.
        strategy='best1bin',
        #recombination high and mutation fixed (not dithered) since there's no ruggedness to hedge against.
        recombination=0.9,
        mutation=0.6    # dithering (0.5, 1.0) adds step-size variation to escape local optima
    )

    print(f"Team {t} coefficients are {result.x} obj: {result.fun} evaluations {result.nfev} - {result.success}")
    return t, result.x, result.success, result.fun, result.nit, result.message


def calibration_objective(coefficients_c, context):
    """Reset the selected trial before every objective evaluation.

    Differential Evolution calls this function many times for each team. The
    stock generator mutates shared module state while evaluating a coefficient
    set, so the trial inputs must be rebuilt before every call. pkl_fs doesn't
    depend on coefficients_c, so it's loaded once per team (see
    run_calibration_for_team) and reused here rather than re-read from disk.
    """
    reset_trial_inputs(
        context["trial"],
        context["exp_data"],
        context["trial_pinp"],
        context["sinp_defaults"],
        context["uinp_defaults"],
        context["pinp_defaults"],
        user_sa=context["user_sa"],
    )

    ##an extreme coefficient set can make the generator fail (e.g. so many animals die that the w axis can't be
    ## collapsed). Return a large objective (as for a NaN objective in the generator) so DE discards that point,
    ## rather than the error ending the calibration of every team.
    try:
        objective = sgen.generator(
            coefficients_c=coefficients_c,
            params={},
            r_vals={},
            nv={},
            pkl_fs_info={},
            pkl_fs=context["pkl_fs"],
            stubble=context["stubble"],
            calibrate_trait_values=RUN_CALIBRATION,
            calibration_weights_p=context["calibration_weights"],
            calibration_targets_p=context["calibration_targets"],
            calibration_trait_names_p=context["trait_names_p"],
        )
    except Exception as e:
        print(f"Warning: the generator failed for Team {context.get('team')} so the objective is set to 1e8."
              f" {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return 1e8
    ##track the best objective for the step report (see f1_report_step in run_calibration_for_team)
    if "progress" in context:
        context["progress"]["best"] = min(context["progress"]["best"], objective)
    return objective


def calibration_traits(coefficients_c, context):
    """Trait values and objective for one coefficient set (as calibration_objective, but also returning the trait values,
    which the least squares method needs). Returns (None, 1e8) if the generator fails or the objective is NaN."""
    reset_trial_inputs(context["trial"], context["exp_data"], context["trial_pinp"], context["sinp_defaults"]
                       , context["uinp_defaults"], context["pinp_defaults"], user_sa=context["user_sa"])
    out = {}
    try:
        objective = sgen.generator(coefficients_c=coefficients_c, params={}, r_vals={}, nv={}, pkl_fs_info={}
                                   , pkl_fs=context["pkl_fs"], stubble=context["stubble"], calibrate_trait_values=True
                                   , calibration_weights_p=context["calibration_weights"]
                                   , calibration_targets_p=context["calibration_targets"]
                                   , calibration_trait_names_p=context["trait_names_p"], o_calibration_traits=out)
    except Exception as e:
        print(f"Warning: the generator failed for Team {context.get('team')}. {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return None, 1e8
    traits = out.get("output")
    if traits is None or objective >= 1e8 or not np.all(np.isfinite(traits)):
        return None, 1e8
    return np.asarray(traits, dtype=float).ravel(), objective


def f1_jacobian(x, traits_x, lo, up, context, half=None):
    """Forward-difference Jacobian of the trait values w.r.t. the coefficients at x (traits x coefficients).
    Each step is LSQ_STEP of the coefficient's half-range (half, default the half-range of lo/up), towards the further
    bound so the point stays in the bounds (shortened if that bound is nearer than the step).
    Returns the Jacobian and the number of generator evaluations."""
    half = (up - lo) / 2 if half is None else half
    jac = np.zeros((traits_x.size, x.size))
    n = 0
    for j in np.flatnonzero((half > 0) & (up > lo)):
        room = max(up[j] - x[j], x[j] - lo[j])
        d = min(LSQ_STEP * half[j], room) * (1 if up[j] - x[j] >= x[j] - lo[j] else -1)
        xj = x.copy()
        xj[j] += d
        traits_j, _ = calibration_traits(xj, context)
        n += 1
        if traits_j is not None:
            jac[:, j] = (traits_j - traits_x) / d
    return jac, n


def f1_standard_traits_worker(x):
    """Pool worker for f_standard_jacobian: trait values at x (team 0's context; the traits don't depend on the targets)."""
    return calibration_traits(x, f1_team_context(0, worker_context))[0]


def f_standard_jacobian(context, n_processes):
    """Jacobian of the trait values at the Standard (all coefficients 0, because they are adjustments to the standard
    values), by central differences with steps of 2 x LSQ_STEP of team 0's half-ranges. Measured once and shared by
    the teams as the starting Jacobian for lsq. The 2 x n_coef evaluations are spread over the processes.
    A step that would cross a cLimit is shortened on that side (an uneven central difference)."""
    lo, up = context["bnd_lo_tc"][0], context["bnd_up_tc"][0]
    h = 2 * LSQ_STEP * (up - lo) / 2
    hp, hm = h.copy(), h.copy()
    if context.get("lim_lo_c") is not None:
        hp, hm = np.clip(context["lim_up_c"], 0, h), np.clip(-context["lim_lo_c"], 0, h)
    free = np.flatnonzero((h > 0) & (hp + hm > 0))
    points = [(hp[j] if s == 1 else -hm[j]) * np.eye(h.size)[j] for j in free for s in (1, -1)]
    if n_processes > 1:
        with mp.Pool(processes=n_processes, initializer=init_worker, initargs=(context.copy(),)) as pool:
            traits = pool.map(f1_standard_traits_worker, points, chunksize=1)
    else:
        init_worker(context.copy())
        traits = [f1_standard_traits_worker(x) for x in points]
    if any(tr is None for tr in traits):
        raise ValueError("The generator failed at a Standard Jacobian point: reduce LSQ_STEP or check the bounds of team 0.")
    jac = np.zeros((traits[0].size, h.size))
    for k, j in enumerate(free):
        jac[:, j] = (traits[2 * k] - traits[2 * k + 1]) / (hp[j] + hm[j])
    return jac


def f1_run_lsq(t, team_context, x0, lo, up, half=None):
    """Bounded Gauss-Newton least squares for one team (see METHOD).

    Minimises sum(((traits - targets) * weights)^2), the same objective as DE. Each step solves the linearised problem
    within the bounds and a trust region (the largest change of any coefficient, in half-ranges). A step that reduces
    the objective is accepted and the trust region grows if the reduction was as predicted (shrinks if much less);
    a step that doesn't is rejected and the trust region shrinks. After each step the Jacobian is updated along the
    step (Broyden), and it is re-measured at the current point when the step failed to halve the objective.
    lo/up are the bounds (the cLimits, which can be infinite) and half the scale (the Low/High half-range); without
    half the scale is the half-range of lo/up."""
    w = np.asarray(team_context["calibration_weights"], dtype=float).ravel()
    tgt = np.asarray(team_context["calibration_targets"], dtype=float).ravel()
    lo, up = np.asarray(lo, dtype=float), np.asarray(up, dtype=float)
    half = (up - lo) / 2 if half is None else np.asarray(half, dtype=float)
    free = (half > 0) & (up > lo)
    progress = team_context["progress"]
    x = np.clip(np.asarray(x0, dtype=float), lo, up)
    traits, f = calibration_traits(x, team_context)
    nfev, n_jac = 1, 0
    if traits is None:
        print(f"Team {t} coefficients are {x} obj: {f} evaluations {nfev} - False")
        return t, x, False, f, 0, "lsq: the generator failed at BestBet"
    r = (traits - tgt) * w
    if team_context.get("jac0") is not None:
        a = team_context["jac0"] * w[:, None]
        a_at_x = False
    else:
        jac, n = f1_jacobian(x, traits, lo, up, team_context, half)
        a, a_at_x, nfev, n_jac = jac * w[:, None], True, nfev + n, n_jac + 1
    radius = 1.0
    while f > LSQ_TARGET and nfev < LSQ_MAX_EVAL and radius > 1e-6:
        ##step: least squares of the linearised weighted errors, within the bounds and the trust region (scaled by the half-ranges)
        lb = np.maximum(lo - x, -radius * half)[free] / half[free]
        ub = np.minimum(up - x, radius * half)[free] / half[free]
        sol = spo.lsq_linear(a[:, free] * half[free], -r, bounds=(lb, np.maximum(ub, lb + 1e-15)), method='bvls')
        p = np.zeros_like(x)
        p[free] = sol.x * half[free]
        x_new = np.clip(x + p, lo, up)
        p = x_new - x    #the step taken (the clip only removes rounding beyond the bounds)
        predicted = np.sum((r + a @ p) ** 2)
        traits_new, f_new = calibration_traits(x_new, team_context)
        nfev += 1
        if traits_new is not None and f_new < f:
            r_new = (traits_new - tgt) * w
            ##Broyden update of the weighted Jacobian along the step
            a += np.outer(r_new - r - a @ p, p) / (p @ p)
            ratio = (f - f_new) / max(f - predicted, 1e-300)
            if ratio > 0.75 and np.max(np.abs(p[free]) / half[free]) > 0.99 * radius:
                radius *= 2
            elif ratio < 0.25:
                radius *= 0.5
            remeasure = f_new > 0.5 * f
            x, traits, r, f, a_at_x = x_new, traits_new, r_new, f_new, False
        else:
            radius *= 0.25
            remeasure = True
        ##the Jacobian in use isn't good enough here (e.g. a curved trait away from the Standard): re-measure it at x
        if remeasure and not a_at_x:
            jac, n = f1_jacobian(x, traits, lo, up, team_context, half)
            a, a_at_x, nfev, n_jac = jac * w[:, None], True, nfev + n, n_jac + 1
        progress["step"] += 1
        print(f"Team {t} step {progress['step']}: f(x)= {f:g} evaluations= {nfev} radius= {radius:.3g} jacobians= {n_jac}\n"
              f"Team {t} step {progress['step']} x: {x.tolist()}\n", end='')
    success = f <= LSQ_DE_ABOVE
    reason = ("reached LSQ_TARGET" if f <= LSQ_TARGET else "LSQ_MAX_EVAL evaluations" if nfev >= LSQ_MAX_EVAL
              else "no further improvement (trust region collapsed)")
    print(f"Team {t} coefficients are {x} obj: {f} evaluations {nfev} - {success}")
    return t, x, success, f, progress["step"], f"lsq: {reason}; {nfev} evaluations, {n_jac} Jacobians re-measured"


def run_calibration_for_team_worker(t):
    """Pool.map wrapper.

    multiprocessing.Pool.map can only pass the team number here. The larger
    context is stored once per worker by init_worker().
    """
    return run_calibration_for_team(t, worker_context)


def build_context(targets_tp, weights_p, bestbet_tc, bnd_lo_tc, bnd_up_tc, n_coef, *, team_processes
                  , trait_names_p, trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults
                  , lim_lo_c=None, lim_up_c=None):
    """Collect the values needed by team calibration and objective evaluation.

    Requires sen to be populated for the current trial (see reset_trial_inputs) because the DE polish, tol
    and atol can be overridden by the sav in exp.xlsx.
    """
    popsize = POPSIZE
    population = popsize * n_coef
    workers = 1 if team_processes else min(mp.cpu_count(), population, MAX_WORKERS_WITHIN_TEAM)
    updating = 'deferred' if workers != 1 else 'immediate'
    ##DE's own step report (disp) doesn't include the team. With workers = 1 the steps are reported with the team by
    ## f1_report_step, so disp is only needed when workers > 1 (f1_report_step can't track the objective)
    disp = workers != 1
    polish = bool(fun.f_sa(POLISH, sen.sav['calib_polish'], 5))
    tol = float(fun.f_sa(TOL, sen.sav['calib_tol'], 5))
    atol = float(fun.f_sa(ATOL, sen.sav['calib_atol'], 5))
    ##calibration method: command line (4th argument) > sav in exp.xlsx > METHOD
    method = str(sys.argv[4] if len(sys.argv) > 4 else fun.f_sa(METHOD, sen.sav['calib_method'], 5)).strip().lower()
    if method not in ('de', 'lsq', 'lsq+de'):
        raise ValueError(f"Unknown calibration method '{method}'. Use 'de', 'lsq' or 'lsq+de'.")
    return {
        "method": method,
        "targets_tp": targets_tp,
        "weights_p": weights_p,
        "trait_names_p": trait_names_p,
        "bestbet_tc": bestbet_tc,
        "bnd_lo_tc": bnd_lo_tc,
        "bnd_up_tc": bnd_up_tc,
        "lim_lo_c": lim_lo_c,
        "lim_up_c": lim_up_c,
        "trial": trial,
        "exp_data": exp_data,
        "trial_pinp": trial_pinp,
        "sinp_defaults": sinp_defaults,
        "uinp_defaults": uinp_defaults,
        "pinp_defaults": pinp_defaults,
        "stubble": False,
        "maxiter": MAXITER,
        "popsize": popsize,
        "tol": tol,
        "atol": atol,
        "disp": disp,
        "polish": polish,
        "workers": workers,
        "updating": updating,
    }


def write_calibration_results(df_targets_tp, df_bnd_lo_tc, df_bnd_up_tc, keys_t, keys_c, results, n_teams_total=None
                              , wait_if_open=True, df_weights_p=None):
    n_teams = len(keys_t) if n_teams_total is None else n_teams_total
    n_coef = len(keys_c)
    coefficients_tc = np.zeros((n_teams, n_coef))
    success_t = np.zeros(n_teams, dtype=bool)
    wsmse_t = np.full(n_teams, np.nan)
    nit_t = np.zeros(n_teams)
    message_t = np.empty(n_teams, dtype=object)
    message_t[:] = "Pending"
    traits_tp = np.full((n_teams, df_targets_tp.shape[1]), np.nan)

    for t, coefficients, success, wsmse, nit, message, *traits in sorted(results, key=lambda item: item[0]):
        coefficients_tc[t, :] = coefficients
        success_t[t] = success
        wsmse_t[t] = wsmse
        nit_t[t] = nit
        message_t[t] = message
        if traits and traits[0] is not None:
            traits_tp[t, :] = traits[0]

    ##trait values at the final coefficients, the misses (achieved - target) and each trait's share of the objective
    ## ((miss x weight)^2). The Traits sheet has the same layout as Output/TraitValues.xlsx (teams in column A).
    traits = pd.DataFrame(traits_tp, index=keys_t, columns=df_targets_tp.columns)
    misses = traits - df_targets_tp.values
    contributions = None
    if df_weights_p is not None:
        contributions = (misses * np.asarray(df_weights_p, dtype=float).ravel()) ** 2
        contributions.insert(0, "Objective (sum)", contributions.sum(axis=1, min_count=1))

    coefficients = pd.DataFrame(coefficients_tc, index=keys_t, columns=keys_c)
    success = pd.DataFrame(success_t, index=keys_t, columns=["Optimal"])
    wsmse  = pd.DataFrame(wsmse_t, index=keys_t, columns=["WSMSE"])
    nit = pd.DataFrame(nit_t, index=keys_t, columns=["Iterations"])
    message = pd.DataFrame(message_t, index=keys_t, columns=["Message"])

    ### Write to Excel
    ###if the file is open in Excel: the outcome depends on the flag 'wait if open'.
    ### skip if False (the next write includes all results so far), otherwise wait for the user to close Excel
    calibration_path = relativeFile.findExcel('CalibrationResults.xlsx')
    writer = fun.f_excel_writer(calibration_path, wait_if_open)
    if writer is None:
        print(f"Warning: could not write {calibration_path} (probably open in Excel). It will be re-written when the next team finishes.")
        return False
    with writer:
        df_targets_tp.to_excel(writer, "Targets", index=True, header=True, startrow=0, startcol=1)
        coefficients.to_excel(writer, "Coefficients", index=True, header=True, startrow=0, startcol=1)
        success.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+2)
        wsmse.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+3)
        nit.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+4)
        message.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+5)
        df_bnd_lo_tc.to_excel(writer, "Low", index=True, header=True, startrow=0, startcol=1)
        df_bnd_up_tc.to_excel(writer, "High", index=True, header=True, startrow=0, startcol=1)
        traits.to_excel(writer, "Traits", index=True, header=True)
        misses.to_excel(writer, "Misses", index=True, header=True)
        if contributions is not None:
            contributions.to_excel(writer, "Contributions", index=True, header=True)
    return True


def main():
    #report the clock time that the experiment was started
    print(f'Calibration commenced at: {time.ctime()}')
    time_list = [timer()]

    # Handle command line arguments
    global RUN_CALIBRATION
    try:
        if len(sys.argv) > 3:
            RUN_CALIBRATION = (sys.argv[3].lower() in ['true', '1', 'yes'])
    except:
        pass

    ##load excel data and experiment data
    exp_data, exp_group_bool, trial_pinp = exp.f_read_exp()
    exp_data = exp.f_group_exp(exp_data, exp_group_bool)
    dataset = list(np.flatnonzero(np.nan_to_num(np.array(exp_data.index.get_level_values(0))))) # Define the dataset - trials that require running
    sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)

    if len(dataset) > 1 and RUN_CALIBRATION:
        raise ValueError("Can't run calibration with multiple trials. Select an experiment with one active trial.")

    o_trait_values = {}

    ##loop through trials. there can only be one active trial if running the calibration (can be multiple if reporting)
    for trial in dataset:
        o_trait_values[trial] = {}  # create row key inside calibration dictionary

        df_targets_tp, df_weights_p, df_bestbet_tc, df_bnd_lo_tc, df_bnd_up_tc, df_lim_c = read_calibration_control()
        check_calibration_control(df_targets_tp, df_weights_p, df_bestbet_tc, df_bnd_lo_tc, df_bnd_up_tc, df_lim_c)
        lim_lo_c, lim_up_c = f1_limits(df_lim_c) if df_lim_c is not None else (None, None)

        keys_t = df_targets_tp.index
        keys_c = df_bestbet_tc.columns
        n_coef = len(keys_c)
        n_teams = len(keys_t)

        ##processors
        ## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
        try:
            maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the trial is the first value hence take the second.
        except IndexError:  # in case no arg passed to python
            maximum_processes = 1  # available memory / value determined by size of the model being run (~5GB for the small model)
        ## number of agents (processes) should be min of the num of cpus, number of teams or the user specified limit due to memory capacity
        n_processes = min(mp.cpu_count(), n_teams, maximum_processes)

        ###convert to np
        targets_tp = df_targets_tp.values
        weights_p = df_weights_p.values
        bestbet_tc = df_bestbet_tc.values
        bnd_lo_tc = df_bnd_lo_tc.values
        bnd_up_tc = df_bnd_up_tc.values

        team_processes = n_processes != 1
        ##populate sen for the current trial so build_context can apply the DE control sav
        reset_trial_inputs(trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults)
        context = build_context(
            targets_tp,
            weights_p,
            bestbet_tc,
            bnd_lo_tc,
            bnd_up_tc,
            n_coef,
            team_processes=team_processes,
            trait_names_p=list(df_targets_tp.columns),
            trial=trial,
            exp_data=exp_data,
            trial_pinp=trial_pinp,
            sinp_defaults=sinp_defaults,
            uinp_defaults=uinp_defaults,
            pinp_defaults=pinp_defaults,
            lim_lo_c=lim_lo_c,
            lim_up_c=lim_up_c
        )

        teams = list(range(n_teams))
        if RUN_CALIBRATION:
            print(f"Calibration method: {context['method']}")
            print(f"DE controls: polish={context['polish']} tol={context['tol']} atol={context['atol']}")
            if context["method"] != 'de':
                print(f"lsq controls: step={LSQ_STEP} target={LSQ_TARGET} max evaluations={LSQ_MAX_EVAL} DE above={LSQ_DE_ABOVE}")
                print("lsq bounds: " + ("cLimits (scaled by Low/High)" if lim_lo_c is not None else "Low/High (no cLimits sheet)"))
                t0 = timer()
                context["jac0"] = f_standard_jacobian(context, n_processes)
                print(f"Jacobian at the Standard measured in {timer() - t0:.0f} s")
            if team_processes:
                print(f"multiprocess across {n_processes} teams with population of {context['popsize'] * n_coef}")
                worker_init_context = context.copy()
                # The worker context contains the trial reset inputs used by calibration_objective().
                # build_context forces workers=1 per team whenever team_processes is True, so DE never tries to
                # spin up its own nested Pool from inside one of these (daemonic) worker processes - daemonic
                # processes aren't allowed to have children, so that invariant must hold for this to stay safe.
                with mp.Pool(processes=n_processes,
                             initializer=init_worker,
                             initargs=(worker_init_context,)) as pool:
                    results = []
                    try:
                        for result in pool.imap_unordered(run_calibration_for_team_worker, teams, chunksize=1):
                            results.append(result)
                            write_calibration_results(df_targets_tp, df_bnd_lo_tc, df_bnd_up_tc, keys_t, keys_c
                                                      , results, n_teams_total=n_teams, wait_if_open=False, df_weights_p=df_weights_p)
                    except KeyboardInterrupt:
                        print("\nKeyboardInterrupt received — terminating worker processes...")
                        pool.terminate()
                        pool.join()
                        raise
            else:
                print(f"multiprocess the population of {context['popsize'] * n_coef} with {context['workers']} workers")
                mp.freeze_support()
                results = []
                for t in teams:
                    result = run_calibration_for_team(t, context)
                    results.append(result)
                    time_list.append(timer())
                    print(f"Team {t} time {time_list[-1] - time_list[-2]:0.4f}secs")
                    write_calibration_results(df_targets_tp, df_bnd_lo_tc, df_bnd_up_tc, keys_t, keys_c
                                              , results, n_teams_total=n_teams, wait_if_open=False, df_weights_p=df_weights_p)

            write_calibration_results(df_targets_tp, df_bnd_lo_tc, df_bnd_up_tc, keys_t, keys_c
                                      , results, wait_if_open=True, df_weights_p=df_weights_p)

            time_list.append(timer())
            time_elapsed = time_list[-1] - time_list[0]
            print(f"elapsed total time for calibration {time_elapsed//3600:>02.0f}:{time_elapsed%3600//60:02.0f}:{time_elapsed%60:07.4f} ") # Time in seconds

        else: #if just reporting trait values
            for t in teams:
                o_trait_values[trial][t] = {}  # create team key inside calibration traits dictionary
                pkl_fs = setup_trial(
                    context["trial"],
                    context["exp_data"],
                    context["trial_pinp"],
                    context["sinp_defaults"],
                    context["uinp_defaults"],
                    context["pinp_defaults"],
                )

                sgen.generator(
                    coefficients_c=context["bestbet_tc"][t],
                    params={},
                    r_vals={},
                    nv={},
                    pkl_fs_info={},
                    pkl_fs=pkl_fs,
                    o_trait_values=o_trait_values[trial][t],
                    stubble = context["stubble"],
                    calibration_weights_p = context["weights_p"],
                    calibration_targets_p = context["targets_tp"][t],
                    calibration_trait_names_p = context["trait_names_p"],
                )

    ## this is for saving the calibration trait values for each team.
    if not RUN_CALIBRATION:
        print("Saving trait values...")
        data = {}
        for trial, team_dict in o_trait_values.items():
            for t, team_data in team_dict.items():
                key = f"Trial_{trial}_Team_{t}"
                if isinstance(team_data, dict) and "output" in team_data:
                    data[key] = team_data["output"]
                else:
                    print(f"Warning: No output for Trial {trial} Team {t}")

        if data:
            traits_to_save = pd.DataFrame(data, index=df_targets_tp.columns).T   #trait names as the column headings
            trait_file_path = relativeFile.find(__file__, "../../Output", "TraitValues.xlsx")
            with fun.f_excel_writer(trait_file_path) as writer:
                traits_to_save.to_excel(writer, sheet_name='Traits', index=True, header=True)
            print(f'Trait values written to {trait_file_path}. Note: Optimisation is not being carried out')
        else:
            print("No trait data was collected to save.")

if __name__ == '__main__':
    main()

