import time
import math
import os.path
from datetime import datetime
import multiprocessing
import sys

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
force_run = False #set to True if you want to force all trials to run even if they are up to date.
solver_method = 'CPLEX'
print_debug_output = False

########################################
##load excel data and experiment data  #
########################################
exp_data, exp_data1, dataset, trial_pinp, total_trials = exp.f_load_experiment_data(force_run)
sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)
d_rot_info = dxl.f_load_phases()
cat_propn_s1_ks2 = dxl.f_load_stubble()

##############
##processors #
##############
## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
try:
    maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
except IndexError:  # in case no arg passed to python
    maximum_processes = 1  # available memory / value determined by size of the model being run (~5GB for the small model)
## number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
n_processes = min(multiprocessing.cpu_count(),len(dataset),maximum_processes)

###########
##run AFO #
###########
## set the start time at the beginning of the multiprocessing loops
start_loops_time = time.time()    #This excludes the time for reading the inputs and is used to calculate remaining time
def run_afo(row):
    ##start timer for each loop
    start_trial_time = time.time()

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][3]
    trial_description = f'{dataset.index(row)+1} {trial_name}'
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
    global d_rot_info #has to be defined as global sincie it is defined outside this function above
    model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info = (
        afo.exp(solver_method, user_sa, property, trial_name, trial_description, sinp_defaults, uinp_defaults,
                pinp_defaults, d_rot_info, cat_propn_s1_ks2, pkl_fs, print_debug_output))

    ##save AFO outputs
    out.f_save_trial_outputs(exp_data, row, trial_name, model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info)

    #last step is to print the time for the current trial to run
    ##determine expected time to completion - trials left multiplied by average time per trial
    ##this approach works well if chunksize = 1
    total_batches = math.ceil(len(dataset) / n_processes)
    current_batch = math.ceil((dataset.index(row)+1) / n_processes) #add 1 because python starts at 0
    remaining_batches = total_batches - current_batch
    time_taken = time.time() - start_loops_time   #since start of the multiprocessing loops (excludes reading inputs)
    batch_time = time_taken / current_batch
    time_remaining = remaining_batches * batch_time
    finish_time_expected = time.time() + time_remaining
    trial_time = time.time() - start_trial_time

    print(f'{trial_description}, total time taken this trial: {trial_time:.2f}')
    message = f'{trial_description}, Expected finish time: \033[1m{time.ctime(finish_time_expected)}\033[0m at {time.ctime()}'
    #replace message if this process is complete
    if remaining_batches == 0:
        message = f'{trial_description}, this process is complete'
    print(message)




##works when run through anaconda prompt - if 9 runs and 8 processors, the first processor to finish, will start the 9th run
#   using map it returns outputs in the order they go in ie in the order of the exp
##the result after the different processes are done is a list of dicts (because each iteration returns a dict and the multiprocess stuff returns a list)
if __name__ == '__main__':
    #todo could intercept if len(dataset) == 0 which leads to an error message
    # if there are no trials it would also be good to report whether there are no trials (because of user error in specifying the trial number) or if they exist but don't need running
    ##start multiprocessing
    with multiprocessing.Pool(processes=n_processes) as pool:
        ##size 1 has similar speed even for N11 model and allows better reporting (will be even better on a larger model)
        ##a drawback of chunksize = 1 is that if there is an error in the multiprocessed code then every trial is still processed
        trials_successfully_run = pool.map(run_afo, dataset, chunksize = 1)
    end = time.time()
    print(f'\n\033[1mExperiment completed at:\033[0m {time.ctime()}, total time taken: {end - start_time:.2f}')
    try:
        print(f'average time taken for each loop: {(end - start_time) / len(dataset):.2f}')  #average time since start of experiment
    except ZeroDivisionError:
        pass
