# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John

SheepCalibration.py is run using python -u SheepCalibration.py {<exp no>} {<number of processes>}
<exp no> is an optional argument, if it is excluded the default trial is QT
<number of processes> is an optional argument. The default is 1 (don't multiprocess teams use workers on a single team)

Multiprocessing the teams (with workers =1) will be quicker if
 (1) the number of cpus is greater than population size - Population size is pop_size parameter (5 to 15) x number of coefficients
 (2) there are sufficient teams to occupy the computer resource
If not, use multiple workers. The maximum useful number of workers is the size of the selection population
Multiprocessing teams can be more efficient because it can use 'immediate' updating

sys.argv: Experiment number (will use the first trial in the experiment). If blank uses QT (trial 12)
          Number of multi processes. If blank will not multiprocess but will use workers
"""


import numpy as np
from timeit import default_timer as timer
import sys
import os

##Calibration specific imports
import pandas as pd
from scipy import optimize as spo
import multiprocessing as mp
import multiprocessing
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

# from lib.AfoLogic import relativeFile

###############
#User control #
###############
RUN_CALIBRATION = True #set to False if you want to report the trait values for a given trial.

# Optimisation controls. Team multiprocessing always uses one worker per team;
# single-team mode can use workers within scipy's differential_evolution.
MAXITER = 400
POPSIZE = 5
MAX_WORKERS_WITHIN_TEAM = 1
TOL = 0.01
DISP = True
POLISH = False

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
    return df_targets_tp, df_weights_p, df_bestbet_tc, df_bnd_lo_tc, df_bnd_up_tc


def init_worker(context):
    """Store per-trial data in each team worker.

    The actual trial reset happens inside calibration_objective() before every
    sgen.generator call. This initializer only gives the worker enough context
    to run multiple teams without re-pickling that context for every task.
    """
    context = context.copy()
    worker_context.clear()
    worker_context.update(context)


def run_calibration_for_team(t, context):
    """Call Differential Evolution for one team.

    This is used by both run modes:
    - sequential loop over teams
    - multiprocessing loop where each process handles a team
    """
    bounds = list(zip(context["bnd_lo_tc"][t], context["bnd_up_tc"][t]))
    bestbet = context["bestbet_tc"][t]
    team_context = context.copy()
    team_context["team"] = t
    team_context["calibration_weights"] = context["weights_p"]
    team_context["calibration_targets"] = context["targets_tp"][t]

    result = spo.differential_evolution(calibration_objective, bounds
        , args = (team_context,)
        , maxiter=context["maxiter"], popsize=context["popsize"], tol=context["tol"], disp=context["disp"]
        , polish=context["polish"], updating=context["updating"], workers=context["workers"], x0=bestbet)
    print(f"Team {t} coefficients are {result.x} obj: {result.fun} evaluations {result.nfev}")
    return t, result.x, result.success, result.fun, result.nit, result.message


def calibration_objective(coefficients_c, context):
    """Reset the selected trial before every objective evaluation.

    Differential Evolution calls this function many times for each team. The
    stock generator mutates shared module state while evaluating a coefficient
    set, so the trial inputs must be rebuilt before every call.
    """
    pkl_fs = setup_trial(
        context["trial"],
        context["exp_data"],
        context["trial_pinp"],
        context["sinp_defaults"],
        context["uinp_defaults"],
        context["pinp_defaults"],
    )

    return sgen.generator(
        coefficients_c=coefficients_c,
        params={},
        r_vals={},
        nv={},
        pkl_fs_info={},
        pkl_fs=pkl_fs,
        stubble=context["stubble"],
        calibrate_trait_values=RUN_CALIBRATION,
        calibration_weights_p=context["calibration_weights"],
        calibration_targets_p=context["calibration_targets"],
    )


def run_calibration_for_team_worker(t):
    """Pool.map wrapper.

    multiprocessing.Pool.map can only pass the team number here. The larger
    context is stored once per worker by init_worker().
    """
    return run_calibration_for_team(t, worker_context)


def build_context(targets_tp, weights_p, bestbet_tc, bnd_lo_tc, bnd_up_tc, n_coef, *, team_processes
                  , trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults):
    """Collect the values needed by team calibration and objective evaluation."""
    popsize = POPSIZE
    population = popsize * n_coef
    workers = 1 if team_processes else min(multiprocessing.cpu_count(), population, MAX_WORKERS_WITHIN_TEAM)
    updating = 'deferred' if workers != 1 else 'immediate'
    return {
        "targets_tp": targets_tp,
        "weights_p": weights_p,
        "bestbet_tc": bestbet_tc,
        "bnd_lo_tc": bnd_lo_tc,
        "bnd_up_tc": bnd_up_tc,
        "trial": trial,
        "exp_data": exp_data,
        "trial_pinp": trial_pinp,
        "sinp_defaults": sinp_defaults,
        "uinp_defaults": uinp_defaults,
        "pinp_defaults": pinp_defaults,
        "stubble": False,
        "maxiter": MAXITER,
        "popsize": popsize,
        "tol": TOL,
        "disp": DISP,
        "polish": POLISH,
        "workers": workers,
        "updating": updating,
    }


def write_calibration_results(df_targets_tp, df_bnd_lo_tc, df_bnd_up_tc, keys_t, keys_c, results):
    n_teams = len(keys_t)
    n_coef = len(keys_c)
    coefficients_tc = np.zeros((n_teams, n_coef))
    success_t = np.zeros(n_teams, dtype=bool)
    wsmse_t = np.zeros(n_teams)
    nit_t = np.zeros(n_teams)
    message_t = np.empty(n_teams, dtype=object)

    for t, coefficients, success, wsmse, nit, message in sorted(results, key=lambda item: item[0]):
        coefficients_tc[t, :] = coefficients
        success_t[t] = success
        wsmse_t[t] = wsmse
        nit_t[t] = nit
        message_t[t] = message

    coefficients = pd.DataFrame(coefficients_tc, index=keys_t, columns=keys_c)
    success = pd.DataFrame(success_t, index=keys_t, columns=["Optimal"])
    wsmse  = pd.DataFrame(wsmse_t, index=keys_t, columns=["WSMSE"])
    nit = pd.DataFrame(nit_t, index=keys_t, columns=["Iterations"])
    message = pd.DataFrame(message_t, index=keys_t, columns=["Message"])

    ### Write to Excel
    calibration_path = relativeFile.findExcel('CalibrationResults.xlsx')
    writer = pd.ExcelWriter(calibration_path, engine='xlsxwriter')
    df_targets_tp.to_excel(writer, "Targets", index=True, header=True, startrow=0, startcol=1)
    coefficients.to_excel(writer, "Coefficients", index=True, header=True, startrow=0, startcol=1)
    success.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+2)
    wsmse.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+3)
    nit.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+4)
    message.to_excel(writer, "Coefficients", index=False, header=True, startrow=0, startcol=n_coef+5)
    df_bnd_lo_tc.to_excel(writer, "Low", index=True, header=True, startrow=0, startcol=1)
    df_bnd_up_tc.to_excel(writer, "High", index=True, header=True, startrow=0, startcol=1)
    writer.close()


def main():
    #report the clock time that the experiment was started
    print(f'Calibration commenced at: {time.ctime()}')
    time_list = [timer()]

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

        df_targets_tp, df_weights_p, df_bestbet_tc, df_bnd_lo_tc, df_bnd_up_tc = read_calibration_control()

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
        n_processes = min(multiprocessing.cpu_count(), n_teams, maximum_processes)

        ###convert to np
        targets_tp = df_targets_tp.values
        weights_p = df_weights_p.values
        bestbet_tc = df_bestbet_tc.values
        bnd_lo_tc = df_bnd_lo_tc.values
        bnd_up_tc = df_bnd_up_tc.values

        teams = list(range(n_teams))
        if RUN_CALIBRATION:
            team_processes = n_processes != 1
            context = build_context(
                targets_tp, weights_p, bestbet_tc, bnd_lo_tc, bnd_up_tc, n_coef,
                team_processes=team_processes, trial=trial, exp_data=exp_data, trial_pinp=trial_pinp,
                sinp_defaults=sinp_defaults, uinp_defaults=uinp_defaults, pinp_defaults=pinp_defaults
            )
            if team_processes:
                print(f"multiprocess across {n_processes} teams")
                worker_init_context = context.copy()
                # The worker context contains the trial reset inputs used by calibration_objective().
                with multiprocessing.Pool(processes=n_processes, initializer=init_worker, initargs=(worker_init_context,)) as pool:
                    results = pool.map(run_calibration_for_team_worker, teams, chunksize=1)
            else:
                print(f"multiprocess the population of {context['popsize'] * n_coef} with {context['workers']} workers")
                mp.freeze_support()
                results = []
                for t in teams:
                    result = run_calibration_for_team(t, context)
                    results.append(result)
                    time_list.append(timer())
                    print(f"Team {t} time {time_list[-1] - time_list[-2]:0.4f}secs")

            write_calibration_results(df_targets_tp, df_bnd_lo_tc, df_bnd_up_tc, keys_t, keys_c, results)

            time_list.append(timer())
            time_elapsed = time_list[-1] - time_list[0]
            print(f"elapsed total time for calibration {time_elapsed//3600:>02.0f}:{time_elapsed%3600//60:02.0f}:{time_elapsed%60:07.4f} ") # Time in seconds

        else: #if just reporting trait values
            pkl_fs = setup_trial(trial, exp_data, trial_pinp, sinp_defaults, uinp_defaults, pinp_defaults)
            sgen.generator(params={}, r_vals={}, nv={}, pkl_fs_info={}, pkl_fs=pkl_fs,
                           o_trait_values=o_trait_values[trial])

    ## this is for saving the calibration trait values for each team.
    if not RUN_CALIBRATION:
        traits_to_save = pd.DataFrame({trial: values["output"] for trial, values in o_trait_values.items()}).T
        writer = pd.ExcelWriter("Output/TraitValues.xlsx", engine='xlsxwriter')
        traits_to_save.to_excel(writer, sheet_name='Traits', index=True, header=False)
        writer.close()
        print(f'Trait values written to Excel. Note: Optimisation is not being carried out')


if __name__ == '__main__':
    main()

