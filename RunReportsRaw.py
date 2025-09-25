import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
import glob
import time

from lib.AfoLogic import ReportFunctions as rfun, relativeFile
from lib.AfoLogic import ReportControl as rep
from lib.RawVersion import LoadExp as exp
from lib.RawVersion import RawVersionReportExtras as rve


#report the clock time that the experiment was started
print(f'Reporting commenced at: {time.ctime()}')
start = time.time()

##load report data
##read in the Excel sheet that controls which reports to run and slice for the selected experiment.
## If no arg passed in or the experiment is not set up with custom col in report_run then default col is used
report_run = pd.read_excel(relativeFile.findExcel('exp.xlsx'), sheet_name='RunReport', index_col=[0], header=[0,1], engine='openpyxl')
try:
    exp_group = int(sys.argv[3])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
except:  # in case no arg passed to python
    exp_group = "Default"
try:
    report_run = report_run.loc[:,('Run',exp_group)]
except KeyError:  # in case the experiment is not set up with custom report_run
    report_run = report_run.loc[:,('Run',"Default")]
report_run = report_run.to_frame()
report_run = report_run.droplevel(1, axis=1)

def f_report(processor, trials, non_exist_trials):

    ##get user controls if they exist
    try:
        import ReportControlsUser as rcu
        user_controls = rcu.f_init_user_controls()
    except ModuleNotFoundError:
        user_controls = {}
    ##create empty df that are used to stack reports for each trial
    stacked_reps = rve.f_create_report_dfs(non_exist_trials)

    ##run reports for each trial and stack with other trials
    for trial_name in trials:
        ###run
        lp_vars, r_vals = rfun.load_pkl(trial_name)
        reports = rep.f_run_report(lp_vars,r_vals, report_run, trial_name, user_controls=user_controls)
        ###stack
        stacked_reps = rve.f_concat_reports(stacked_reps, reports, report_run, trial_name)

    ##save to excel
    rve.f_save_reports(report_run, stacked_reps, processor)



##run reports
if __name__ == '__main__':
    ##read in exp log
    exp_data, experiment_trials, trial_pinp = exp.f_read_exp()

    ##check if trial results are up-to-date. Out-dated if:
    ##  1. exp.xls has changed
    ##  2. any python module has been updated
    ##  3. the trial needed to be run last time but the user opted not to run that trial
    exp_data = exp.f_run_required(exp_data, trial_pinp)
    exp_data = exp.f_group_exp(exp_data, experiment_trials)  # cut exp_data based on the experiment group
    trial_outdated = exp_data['run_req']  # returns true if trial is out of date

    ## enter the trials to summarise and the reports to include
    trials = np.array(exp_data.index.get_level_values(3))[
        pd.Series(exp_data.index.get_level_values(2)).fillna(0).astype(
            bool)]  # this is slightly complicated because blank rows in exp.xl result in nan, so nan must be converted to 0.

    ##check the trials you want to run exist and are up-to-date - if trial doesn't exist it is removed from trials to
    # report array so that the others can still be run. A list of trials that don't exist is the 'non_exist' sheet in report excel.
    trials, non_exist_trials = rfun.f_errors(trial_outdated,trials)

    ##clear the old report.xlsx
    reports = relativeFile.find(__file__, "./Output", "Report*.xlsx")
    for f in glob.glob(reports):
        os.remove(f)


    ##print out the reports being run and number of trials
    print('Number of trials to run: ', len(trials))

    ##determine the processor for each report
    ## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
    try:
        maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
    except IndexError:  # in case no arg passed to python
        maximum_processes = 1

    ##start multiprocessing
    ### number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
    agents = min(multiprocessing.cpu_count(), len(trials), maximum_processes)
    ###set up dataset for f_report
    args = []
    for agent in list(range(agents)):
        start_trial = int(len(trials)/agents * agent)
        end_trial = int(len(trials)/agents * (agent+1))
        process_trials = trials[start_trial:end_trial]
        arg = [agent,process_trials, non_exist_trials]
        args.append(arg)
    with multiprocessing.Pool(processes=agents) as pool:
        pool.starmap(f_report, args)

    end = time.time()
    # print("Reports successfully completed")
    print(f'Reporting successfully completed at: {time.ctime()}, total time taken: {end - start:.2f}')
