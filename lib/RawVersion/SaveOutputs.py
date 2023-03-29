import numpy as np
import pandas as pd
import pyomo.environ as pe
import time
import os.path
import pickle as pkl
import warnings

from lib.AfoLogic import Functions as fun
from lib.AfoLogic import PropertyInputs as pinp
from lib.AfoLogic import FeedSupplyStock as fsstk

def f_save_trial_outputs(exp_data, row, trial_name, model, profit, lp_vars, r_vals, pkl_fs_info, d_rot_info):
    ##check Output folders exist for outputs. If not create.
    if os.path.isdir('Output'):
        pass
    else:
        os.mkdir('Output')
    if os.path.isdir('Output/infeasible'):
        pass
    else:
        os.mkdir('Output/infeasible')


    ##This writes variable summary each iteration with generic file name - it is overwritten each iteration and is created so the run progress can be monitored
    fun.write_variablesummary(model, row, exp_data, profit, 1, property_id=pinp.general['i_property_id'])

    ##check if user wants full solution
    if exp_data.index[row][1] == True:
        ##make lp file
        model.write('Output/%s.lp' %trial_name,io_options={'symbolic_solver_labels':True})  #file name has to have capital

        ##This writes variable summary for full solution (same file as the temporary version created above)
        fun.write_variablesummary(model, row, exp_data, profit, property_id=pinp.general['i_property_id'])

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
    with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"wb") as f:
        pkl.dump(lp_vars,f,protocol=pkl.HIGHEST_PROTOCOL)

    ##call function to store optimal feedsupply - do this before r_vals since completion of r_vals trigger successful completion.
    ###Note: A feed supply optimisation can not be carried out with Exp1.py because the trials aren't carried out sequentially
    fsstk.f1_pkl_feedsupply(lp_vars,r_vals,pkl_fs_info)

    ##pickle report values - every time a trial is run (even if pyomo not run)
    ## This has to be last because it controls if the trial needs to be run next time the exp is run (f_run_required)
    with open('pkl/pkl_r_vals_{0}.pkl'.format(trial_name),"wb") as f:
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

    ##start writing
    try:
        writer = pd.ExcelWriter('ExcelInputs/Rotation.xlsx', engine='xlsxwriter')
        ##list of rotations - index: tuple, values: expanded version of rotation
        rot_phases.to_excel(writer, sheet_name='rotation list',index=True,header=False)
        ##con1 - the paramater for which history each rotation provides and requires
        mps_bool_req.to_excel(writer, sheet_name='rotation_req',index=False,header=False)
        mps_bool_prov.to_excel(writer, sheet_name='rotation_prov',index=False,header=False)
        ##con1 set - passed into the pyomo constraint
        rot_hist.to_excel(writer, sheet_name='rotation con1 set',index=True,header=False)
        ##finish writing and save
        writer.save()
    except PermissionError:
        warnings.warn("Warning: Rotation.xlsx open therefore can't save new copy")
