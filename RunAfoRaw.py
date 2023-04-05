import time

from lib.RawVersion import LoadExcelInputs as dxl
from lib.RawVersion import LoadExp as exp
from lib.RawVersion import RawVersionExtras as rve
from lib.AfoLogic import AfoInit as afo
from lib.RawVersion import SaveOutputs as out

start_time = time.time()

############
##controls #
############
force_run = True #set to True if you want to force all trials to run even if they are up to date.

########################################
##load excel data and experiment data  #
########################################
exp_data, exp_data1, dataset, trial_pinp, total_trials = exp.f_load_experiment_data(force_run)
sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs()
d_rot_info = dxl.f_load_phases()
cat_propn_s1_ks2 = dxl.f_load_stubble()

###########
##run AFO #
###########
run = 0  # counter to work out average time per loop
for row in dataset:
    ##start timer for each loop
    start_time = time.time()

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][3]
    trial_description = f'{run} {trial_name}'
    print(f'\n{trial_description}, Starting trial at: {time.ctime()}')

    ##select property for the current trial
    property = trial_pinp.iloc[row]

    ##process user SA
    user_sa = rve.f_process_user_sa(exp_data, row)

    ##run AFO
    model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info = afo.exp(user_sa, property, trial_name, trial_description, sinp_defaults, uinp_defaults, pinp_defaults, d_rot_info, cat_propn_s1_ks2)

    ##tally trials run for print statements
    run += 1

    ##save AFO outputs
    out.f_save_trial_outputs(exp_data, row, trial_name, model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info)

    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    trials_to_go = total_trials - run
    average_time = (time.time() - start_time) / run  # time since the start of experiment
    remaining = trials_to_go * average_time
    finish_time_expected = time.time() + remaining
    print(
        f'{trial_description}, total time taken this loop: {time.time() - start_time:.2f}')  # time since start of this loop
    print(
        f'{trial_description}, Expected finish time: \033[1m{time.ctime(finish_time_expected)}\033[0m (at {time.ctime()})')




end = time.time()
print(f'Experiment completed at: {time.ctime()}, total trials completed: {run}')
try:
    print(f'average time taken for each loop: {(end - start_time) / run:.2f}')  # average time since start of experiment
except ZeroDivisionError:
    pass

