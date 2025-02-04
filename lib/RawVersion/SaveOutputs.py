import numpy as np
import pandas as pd
import pyomo.environ as pe
import time
import os.path
import pickle as pkl
import warnings
import sys

from ..AfoLogic import Functions as fun
from ..AfoLogic import PropertyInputs as pinp
from ..AfoLogic import StructuralInputs as sinp
from ..AfoLogic import FeedSupplyStock as fsstk
from ..AfoLogic import relativeFile
from lib.RawVersion import LoadExcelInputs as dxl
from ..AfoLogic import ReportFunctions as rfun

def f_save_trial_outputs(exp_data, row, trial_name, model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info):
    ##check Output folders exist for outputs. If not create.
    output_path = relativeFile.find(__file__, "../../Output", "")
    output_infeasible_path = relativeFile.find(__file__, "../../Output/infeasible", "")
    if os.path.isdir(output_path):
        pass
    else:
        os.mkdir(output_path)
    if os.path.isdir(output_infeasible_path):
        pass
    else:
        os.mkdir(output_infeasible_path)

    infeasible_trial_file_path = relativeFile.find(__file__, "../../Output/infeasible", trial_name + ".txt")
    if trial_infeasible:
        ###save infeasible file
        with open(infeasible_trial_file_path,'w') as f:
            f.write("Solver error")
    else:  # Something else is wrong - solver may have stalled.
        ###trys to delete the infeasible file because the trial is now optimal
        try:
            os.remove(infeasible_trial_file_path)
        except FileNotFoundError:
            pass


    ##This writes variable summary each iteration with generic file name - it is overwritten each iteration and is created so the run progress can be monitored
    fun.write_variablesummary(model, exp_data.index[row][3], profit, 1, property_id=pinp.general['i_property_id'])

    ##check if user wants full solution
    if exp_data.index[row][1] == True:
        ##make lp file
        model.write('Output/%s.lp' %trial_name,io_options={'symbolic_solver_labels':True})  #file name has to have capital

        if not trial_infeasible:
            ##This writes variable summary for full solution (same file as the temporary version created above)
            fun.write_variablesummary(model, exp_data.index[row][3], profit, property_id=pinp.general['i_property_id'])

            ##prints what you see from pprint to txt file - you can see the slack on constraints but not the rc or dual
            with open('Output/Full model - %s.txt' %trial_name, 'w') as f:  #file name has to have capital
                f.write("My description of the instance!\n")
                model.display(ostream=f)

            ##write rc, duals and slacks to txt file. Duals are slow to write so that option must be turn on
            write_duals = True
            with open('Output/Rc Slacks and Duals - %s.txt' %trial_name,'w') as f:  #file name has to have capital
                f.write('RC\n')
                for v in model.component_objects(pe.Var, active=True):
                    f.write("Variable %s\n" %v)
                    for index in v:
                        try: #in case variable has no index
                            print("      ", index, model.rc[v[index]], file=f)
                        except: pass
                f.write('Slacks (no entry means no slack)\n')  # this can be used in search to find the start of this in the txt file
                for c in model.component_objects(pe.Constraint,active=True):
                    f.write("Constraint %s\n" % c)
                    for index in c:
                        if c[index].lslack() != 0 and c[index].lslack() != np.inf:
                            print("  L   ",index,c[index].lslack(),file=f)
                        if c[index].uslack() != 0 and c[index].lslack() != np.inf:
                            print("  U   ",index,c[index].uslack(),file=f)
                if write_duals:
                    f.write('Dual\n')   #this can be used in search to find the start of this in the txt file
                    for c in model.component_objects(pe.Constraint, active=True):
                        f.write("Constraint %s\n" %c)
                        for index in c:
                            print("      ", index, model.dual[c[index]], file=f)

    ##pickle lp info
    pkl_lp_vars_path = relativeFile.find(__file__, "../../pkl", "pkl_lp_vars_{0}.pkl".format(trial_name))
    with open(pkl_lp_vars_path, "wb") as f:
        pkl.dump(lp_vars,f,protocol=pkl.HIGHEST_PROTOCOL)

    ##call function to store optimal feedsupply - do this before r_vals since completion of r_vals trigger successful completion.
    ###Note: A feed supply optimisation can not be carried out with RunAfoRaw_Multi.py because the trials aren't carried out sequentially
    pkl_fs = fsstk.f1_pkl_feedsupply(lp_vars,r_vals,pkl_fs_info)
    ##store fs if trial is fs_create
    if sinp.structuralsa['i_fs_create_pkl']:
        fs_create_number = sinp.structuralsa['i_fs_create_number']
        # directory_path = os.path.dirname(os.path.abspath(__file__))  # path of directory - required when exp is run from a different location (e.g. in the web app)
        # with open(os.path.join(directory_path, 'pkl/pkl_fs{0}.pkl'.format(fs_create_number)),"wb") as f:
        pkl_fs_path = relativeFile.find(__file__, "../../pkl", 'pkl_fs{0}.pkl'.format(fs_create_number))
        with open(pkl_fs_path, "wb") as f:
            pkl.dump(pkl_fs, f)

    ##pickle report values - every time a trial is run (even if pyomo not run)
    ## This has to be last because it controls if the trial needs to be run next time the exp is run (f_run_required)
    pkl_r_vals_path = relativeFile.find(__file__, "../../pkl", "pkl_r_vals_{0}.pkl".format(trial_name))
    with open(pkl_r_vals_path, "wb") as f:
        pkl.dump(r_vals,f,protocol=pkl.HIGHEST_PROTOCOL)


    ############################################################################################################################################################################################
    ############################################################################################################################################################################################
    #Write rotations and rotation provide/require stuff to excel
    ############################################################################################################################################################################################
    ############################################################################################################################################################################################
    rot_phases = d_rot_info["phases_r"]
    mps_bool_req = d_rot_info["rot_req"]
    mps_bool_prov = d_rot_info["rot_prov"]
    rot_hist = d_rot_info["s_rotcon1"]

    ##load excel version to see if they need to be updated
    xl_d_rot_info = dxl.f_load_phases()
    old_rot_phases = xl_d_rot_info["phases_r"]

    ##start writing
    if not rot_phases.equals(old_rot_phases):
        try:
            rotation_path = relativeFile.findExcel("Rotation.xlsx")
            writer = pd.ExcelWriter(rotation_path, engine='xlsxwriter')
            ##list of rotations - index: tuple, values: expanded version of rotation
            rot_phases.to_excel(writer, sheet_name='rotation list',index=True,header=False)
            ##con1 - the paramater for which history each rotation provides and requires
            mps_bool_req.to_excel(writer, sheet_name='rotation_req',index=False,header=False)
            mps_bool_prov.to_excel(writer, sheet_name='rotation_prov',index=False,header=False)
            ##con1 set - passed into the pyomo constraint
            rot_hist.to_excel(writer, sheet_name='rotation con1 set',index=True,header=False)
            ##finish writing and save
            writer.close()
        except PermissionError:
            warnings.warn("Warning: Rotation.xlsx open therefore can't save new copy")

    ############################################################################################################################################################################################
    ############################################################################################################################################################################################
    #Store results used for model validation before merging to master.
    ############################################################################################################################################################################################
    ############################################################################################################################################################################################
    try:
        exp_group = int(sys.argv[1])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
    except:  # in case no arg passed to python
        exp_group = "Default"

    if exp_group==2:
        ##current trial results - averaged z and q axis
        rfun.f_var_reshape(lp_vars, r_vals)  # this func defines a global variable called d_vars
        trial_sum = pd.concat(rfun.mp_report(lp_vars,r_vals, option=2))

        ##set header to trial name
        trial_name = trial_name.replace(" ", "")[0:15]
        trial_sum.columns = [trial_name]

        ##read in current results and update if trial is one of the validation trials
        try:
            validation_df = pd.read_fwf('AFO Test.txt', index_col=0)
            ###update index incase something new has been added (this mainly occurs for wether sale ages)
            new_idx = validation_df.index.union(trial_sum.index, False)
            validation_df = validation_df.reindex(new_idx)
            ###add or update current trial
            validation_df[trial_name] = trial_sum
        except FileNotFoundError:
            validation_df = trial_sum

        ##save
        with open('AFO Test.txt', 'w') as f:
            f.write(validation_df.to_string())



