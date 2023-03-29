"""

These are inputs that are expected to remain constant between regions and properties, this includes:

* Prices of inputs
* Value of outputs (grain, wool ,meat)
* Interest rates & depreciation rates
* Machinery options
* Sheep parameters and definition of genotypes

author: young
"""

##python modules
import pickle as pkl
import pandas as pd
import numpy as np
import os.path
import sys
import glob
from datetime import datetime

from lib.AfoLogic import Functions as fun
from lib.AfoLogic import Exceptions as exc

def f_read_exp(pinp_req=False):
    '''

    1. Read in exp.xl, set index and cols and drop un-required cols.
    2. Determine which trials are in the experiment the user specified to run.
    3. Determines which property are required for the current exp
    '''

    print('Reading experiment from Excel', end=' ', flush=True)

    ##set the group of trials being run. If no argument is passed in then all trials are run. To pass in argument need to run via terminal.
    try:
        exp_group = int(sys.argv[1]) #reads in as string so need to convert to int, the script path is the first value hence take the second.
    except (IndexError, ValueError) as e: #in case no arg passed to python
        exp_group = None

    ##read excel
    exp_data = pd.read_excel("ExcelInputs/exp.xlsx", index_col=None, header=[0,1,2,3], engine='openpyxl')

    ##determine trials which are in specified experiment group. If no group passed in then all trials will be included in the experiment.
    if exp_group is not None:
        exp_group_bool = exp_data.loc[:,('Drop','blank','blank','Exp Group')].values==exp_group
    else:
        exp_group_bool = exp_data.loc[:,('Drop','blank','blank','Exp Group')].values >= 0 #this will remove the blank rows

    ##Determine which property are required for the current exp
    trial_pinp = exp_data.loc[exp_group_bool, ('Drop', 'blank', 'blank', 'Pinp')]
    if pinp_req:
        print('- finished')
        return trial_pinp.dropna().unique()

    ##drop irrelevant cols and set index
    exp_data = exp_data.iloc[:, exp_data.columns.get_level_values(0)!='Drop']
    exp_data = exp_data.set_index(list(exp_data.columns[0:4]))

    ##get the name of each trial in the experiment group
    experiment_trials = exp_data.index.get_level_values(3)[exp_group_bool]

    ##check if any trials have the same name
    if len(experiment_trials) == len(set(experiment_trials)):
        print('- finished')
    else:
        raise exc.TrialError('''Exp.xl has multiple trials with the same name.''')

    return exp_data, exp_group_bool, trial_pinp

def f_run_required(exp_data1, l_pinp):
    '''
    Here we check if precalcs and pyomo need to be recalculated. this is slightly complicated by the fact that columns and rows can be added to exp.xls
    and the fact that a user can opt not to run a trial even if it is out of date so the run requirement must be tracked
    have any sa cols been added or removed, are the values the same, has the py code changed since last run?

    This function is also used by report.py to calculate if reports are being generated without of date data.

    To trigger trial re-run delete pkl_r_vals{trial_name}.pkl.
    '''
    ##add run cols to be populated - this gets updated during this function and stored for next time.
    ## This tracks if a trial needs to be run but doesnt get run.
    exp_data1['run_req'] = False

    ##try and read in exp from last run - if it doesnt exist then all trials require running.p
    try: #in case pkl_exp doesn't exist, if it doesnt exist then all trials require running
        with open('pkl/pkl_exp.pkl',"rb") as f:
            prev_exp = pkl.load(f)
    except FileNotFoundError:
        exp_data1['run_req']=True #if prev exp doesnt exist then that means all trials require running.
        return exp_data1

    ##calc if any code has been changed since AFO was last run
    ###if only ReportControl.py or ReportFunctions.py have been updated precalcs don't need to be re-run therefore newest is equal to the newest py file that isn't a report
    sorted_list = sorted(glob.iglob('lib/AfoLogic/*.py'), key=os.path.getmtime)
    if sorted_list[-1] != 'ReportFunctions.py' and sorted_list[-1] != 'ReportControl.py':
        newest = sorted_list[-1]
    elif sorted_list[-2] != 'ReportFunctions.py' and sorted_list[-2] != 'ReportControl.py':
        newest = sorted_list[-2]
    else:
        newest = sorted_list[-3]
    same_py = os.path.getmtime('pkl/pkl_exp.pkl') >= os.path.getmtime(newest)

    ##calc if inputs have been changed since AFO was run last (checks all pinp that are used in the exps)
    ###gets the pinp used in the current exp. l_pinp only includes the properties in the current exp but that is fine because the other properties will trigger re-run later.
    l_pinp = l_pinp.dropna().unique()
    same_xl_inputs = True
    for pinp in l_pinp:
        same_xl_inputs &= os.path.getmtime('pkl/pkl_exp.pkl') >= os.path.getmtime(f'ExcelInputs/Property_{pinp}.xlsx')
    same_xl_inputs &= os.path.getmtime('pkl/pkl_exp.pkl') >= os.path.getmtime("ExcelInputs/Universal.xlsx")
    same_xl_inputs &= os.path.getmtime('pkl/pkl_exp.pkl') >= os.path.getmtime("ExcelInputs/Structural.xlsx")

    ##calc if any SA have been added or removed since AFO was last run
    ###get a list of all sa cols (including the name of the trial because two trial may have the same values but have a different name)
    keys_hist = list(prev_exp.reset_index().columns[3:].values)
    keys_current = list(exp_data1.reset_index().columns[3:].values)
    same_SA = keys_current == keys_hist

    ##calc if any SA values have been changed for each trial since AFO was last run (this includes the run_req col which tracks if a trial needed to be run last time but wasn't)
    ###first - update prev_exp run column. this tracks if a trial was run last itteration or if the r_vals were deleted.
    ###if the trial was run the last time the model was run (r_vals are newer than exp.pkl) this trial doesn't need to be re-run unless code or inputs have changed.
    ###if r_vals don't exist the trial needs to be re-run (this allows the user to delete r_vals to re-run a trial).
    run_last = []
    no_r_vals = []
    for trial in prev_exp.index.get_level_values(3):
        try:
            if os.path.getmtime('pkl/pkl_exp.pkl') <= os.path.getmtime('pkl/pkl_r_vals_{0}.pkl'.format(trial)):
                run_last.append(True)
                no_r_vals.append(False)
            else:
                run_last.append(False)
                no_r_vals.append(False)
        except FileNotFoundError:
            run_last.append(False)
            no_r_vals.append(True)
    prev_exp.loc[run_last, ('run_req', '', '', '')] = False #set run req to false if trial was run last iteration of the model.
    prev_exp.loc[no_r_vals, ('run_req', '', '', '')] = True #set run req to True if r_vals don't exist
    ###if the same SA are included, the code is the same and the excel inputs are the same then test if the values in exp.xls are the same
    ### this accounts for the run_req col which tracks if a trial needed running last itteration but was not run.
    if same_SA and same_py and same_xl_inputs:
        ##check if each trial has the same values in exp.xls as last time it was run.
        ## this include the 'run' col which tracks if the exp needed to be run previously and hasnt been.
        i3 = prev_exp.reset_index().set_index(keys_hist).index  # have to reset index because the name of the trial is going to be included in the new index so it must first be dropped from current index
        i4 = exp_data1.reset_index().set_index(keys_current).index
        exp_data1.loc[~i4.isin(i3),('run_req', '', '', '')] = True
    ###if headers are different or py code has changed then all trials need to be re-run
    else: exp_data1['run_req']=True
    return exp_data1

def f_group_exp(exp_data, exp_group_bool):
    '''
    Cuts exp based on the group passed in as argument by user. If no argument then all trials are run.
    This has to be a separate function so that the run required code has access to the full exp.
    '''

    ##cut exp based on group argument
    exp_data = exp_data.loc[exp_group_bool]
    return exp_data

def f_load_experiment_data(force_run):
    '''Read exp.xlsx and determine which trials are being run'''
    ##read in exp.xl and determine which trials are in the experiment group.
    exp_data, exp_group_bool, trial_pinp = f_read_exp()
    exp_data1 = exp_data.copy()  # copy made so that the run col can be added - the original df is used to allocate sa values (would cause an error if run col existed but i can't drop it because it is used to determine if the trial is run)

    ##check if trial needs to be run
    ##trial run if
    ##  1. exp.xls has changed
    ##  2. any python module has been updated
    ##  3. the trial needed to be run last time but the user opted not to run that trial
    exp_data1 = f_run_required(exp_data1, trial_pinp)

    ##check pkl folders exist for outputs. If not create.
    if os.path.isdir('pkl'):
        pass
    else:
        os.mkdir('pkl')

    ##plk a copy of exp. Used next time the model is run to identify which trials are up to date.
    with open('pkl/pkl_exp.pkl', "wb") as f:
        pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)

    ##cut exp_data based on the experiment group
    exp_data = f_group_exp(exp_data, exp_group_bool)
    exp_data1 = f_group_exp(exp_data1, exp_group_bool)

    ## Define the dataset - trials that require running (user wants it run and it is out of date)
    dataset = list(np.flatnonzero(np.nan_to_num(np.array(exp_data.index.get_level_values(0)))
                                  * np.logical_or(force_run, np.array(exp_data1['run_req']))))  # gets the ordinal index values for the trials the user wants to run that are not up to date

    ##print out number of trials to run
    total_trials = sum(exp_data.index[row][0] == True for row in range(len(exp_data)))
    print(f'Number of trials to run: {total_trials}')
    print(f'Number of full solutions: {sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data)))}')
    print(f'exp.xls last saved: {datetime.fromtimestamp(round(os.path.getmtime("ExcelInputs/exp.xlsx")))}')

    return exp_data, exp_data1, dataset, trial_pinp, total_trials


