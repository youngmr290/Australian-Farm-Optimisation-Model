import time

#report the clock time that the experiment was started
print(f'Experiment commenced at: {time.ctime()}')
start_time = time.time()

from lib.RawVersion import LoadExcelInputs as dxl
from lib.RawVersion import LoadExp as exp
from lib.RawVersion import RawVersionExtras as rve
from lib.AfoLogic import AfoInit as afo
from lib.RawVersion import SaveOutputs as out

############
##controls #
############
force_run = True #set to True if you want to force all trials to run even if they are up to date.
solver_method = 'CPLEX'

########################################
##load excel data and experiment data  #
########################################
exp_data, exp_data1, dataset, trial_pinp, total_trials = exp.f_load_experiment_data(force_run)
sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)
d_rot_info = dxl.f_load_phases()
cat_propn_s1_ks2 = dxl.f_load_stubble()

###########
##run AFO #
###########
start_loops_time = time.time()    #This excludes the time for reading the inputs and is used to calculate remaining time

run = 0  # counter to work out average time per loop
for row in dataset:
    ##start timer for each loop
    start_trial_time = time.time()

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][3]
    trial_description = f'{run} {trial_name}'
    print(f'\n{trial_description}, Starting trial at: {time.ctime()}')

    ##select property for the current trial
    property = trial_pinp.iloc[row]

    ##process user SA
    user_sa = rve.f_process_user_sa(exp_data, row)

    ##load pkl_fs based in SA values
    fs_use_pkl = next((item["value"] for item in user_sa if item["key1"] == "fs_use_pkl"), False)
    fs_use_number = next((item["value"] for item in user_sa if item["key1"] == "fs_use_number"), None)
    pkl_fs = dxl.f_load_fs(fs_use_pkl, fs_use_number)

    ##run AFO
    model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info = (
        afo.exp(solver_method, user_sa, property, trial_name, trial_description, sinp_defaults, uinp_defaults,
                pinp_defaults, d_rot_info, cat_propn_s1_ks2, pkl_fs))

    ##tally trials run for print statements
    run += 1

    ##save AFO outputs
    out.f_save_trial_outputs(exp_data, row, trial_name, model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info)

    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    trials_to_go = total_trials - run
    average_time = (time.time() - start_loops_time) / run  # time since the start of the loops (excludes time to read inputs)
    remaining = trials_to_go * average_time
    finish_time_expected = time.time() + remaining
    print(
        f'{trial_description}, time taken this trial: {time.time() - start_trial_time:.2f}')  # time since start of this trial
    print(
        f'{trial_description}, Expected finish time: \033[1m{time.ctime(finish_time_expected)}\033[0m (at {time.ctime()})')




end = time.time()
print(f'Experiment completed at: {time.ctime()}, total trials completed: {run}')
try:
    print(f'average time taken for each loop: {(end - start_time) / run:.2f}')  # average time since start of experiment (including reading inputs)
except ZeroDivisionError:
    pass

