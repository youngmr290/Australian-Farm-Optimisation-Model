'''
Module used to construct lp file from discrete model and stocastic add ons.
'''



import subprocess
from pyomo.core import *
import pyomo.pysp.util.rapper as rapper
import pyomo.pysp.plugins.csvsolutionwriter as csvw
import pyomo.pysp.plugins.jsonsolutionwriter as jsonw
import ReferenceModel as refm
import json

################
#run option    #
################
'''
DSP can be run through a command call or via a rapper module. The rapper module is easier to incorporate in the code 
and hence that is the method used in AFO however the command call method has more options eg generation of .lp.
This module can be used to run the test DSP example from pyomo or the command version of AFO.
'''
run_AFO_command = True #you need to uncomment out the dsp functions in exp.py for this method (i couldn't get it working with them in corepyomo - so essentially need to copy from core to exp then comment out in corepyomo)
run_testDSP_command = False
run_testDSP_rapper = False



################
#run           #
################
##run AFO via terminal command so the lp file is printed have to have the pyomo dsp function in exp for this.
if run_AFO_command:
    subprocess.call(['C:/Users/21512438/Anaconda3/Scripts/runef.exe','--output-file=efout','--solution-writer=pyomo.pysp.plugins.csvsolutionwriter', '--traceback', '--symbolic-solver-labels',
                 '--verbose', '-mExp.py', '--solve', '--output-scenario-tree-solution', '--solver=glpk', '--solver=glpk'])

##option 1 - command runef
##C:/Users/21512438/Anaconda3/Scripts/runef.exe = file for runef
##--output-file=efout = set this to a name saves the lp file (if this argument =false or is not included no lp file will be saved)
##--solution-writer=pyomo.pysp.plugins.jsonsolutionwriter = writes scenario tree to json file
##--solution-writer=pyomo.pysp.plugins.csvsolutionwriter = writes solution to csv file
##--traceback = allows the trace back of any errors
##--symbolic-solver-labels = uses human readable names in the lp (for speed i wonder if it would be better without this option)
##--verbose = prints out extra info
##-mReferenceModel.py = tells runef what model to run
##--solve = tells runef to solve the model (if not then it will just build an lp file)
##--solver=glpk = specifies the solver
##--output-scenario-tree-solution = prints the solution for each element in tree
if run_testDSP_command:
    subprocess.call(['C:/Users/21512438/Anaconda3/Scripts/runef.exe','--output-file=efout','--solution-writer=pyomo.pysp.plugins.csvsolutionwriter', '--traceback', '--symbolic-solver-labels',
                 '--verbose', '-mReferenceModel.py', '--solve', '--output-scenario-tree-solution', '--solver=glpk', '--solver=glpk'])



##option 2 - rapper
if run_testDSP_rapper:
    concrete_tree = refm.pysp_scenario_tree_model_callback()
    stsolver = rapper.StochSolver(None, tree_model=concrete_tree, fsfct=refm.pysp_instance_creation_callback)
    ef_sol = stsolver.solve_ef('glpk')#, tee=True)
    print(ef_sol.solver.termination_condition)
    obj = stsolver.root_E_obj()
    print("Expecatation take over scenarios=", obj)
    for varname, varval in stsolver.root_Var_solution(): # doctest: +SKIP
        print (varname, str(varval))

    ##saves file to csv
    csvw.write_csv_soln(stsolver.scenario_tree,"solution")
    ##saves file to json
    jsonw.JSONSolutionWriter.write('',stsolver.scenario_tree,'ef') #i don't know what the first arg does?? it needs to exist but can put any string without changing output

    ##load json back in
    with open('ef_solution.json') as f:
      data = json.load(f)



# #^to test - Thomas for forum
# def pysp_scenario_tree_model_callback():
#     # Return a NetworkX scenario tree.
#      g = networkx.DiGraph()
#
#     ce1 = "Stage_0"
#     g.add_node("Root",
#            cost=ce1,
#            variables=var_lists[0],
#            derived_variables=[])
#
#     ce2 = "Stage_1"
#     g.add_node("LowPriceScenario",
#            cost=ce2,
#            variables=var_lists[1],
#            derived_variables=[])
#     g.add_edge("Root", "LowPriceScenario", weight=0.999)
#
#     g.add_node("HighPriceScenario",
#            cost=ce2,
#            variables=var_lists[1],
#            derived_variables=[])
#     g.add_edge("Root", "HighPriceScenario", weight=0.001)
#
#     return g
# # the pysp_instance_creation_callback looks like this:
#
# def pysp_instance_creation_callback(self, scenario_name, node_names):
#
#     model = self.model
#     scenario_data = self.scenario_data
#
#     # create instances for different szenarios and write the unique scenariodata into the instance
#     instance = model.clone()   #todo solution time beschleunigen
#     instance.stochastic_data.store_values(scenario_data[scenario_name])
#     return instance
#
# def addStageCost(self,stages_dict,component_dict,model):
#     for stage in stages_dict.keys():
#         if stage == 0:
#             T = range(stages_dict[stage])
#
#         else:
#             T = range(stages_dict[stage - 1],stages_dict[stage])
#
#         model.add_component('Stage_' + str(stage),Expression(expr=cost1 + cost2)))
#
#         return model
#
#     # then I execute the following code:
#
# model = self.addStageCost(stages_dict=stages_dict,component_dict=component_dict,model=model)
#
#
# concrete_tree = self.pysp_scenario_tree_model_callback(stages_dict=stages_dict, component_dict=component_dict, model=model)
# stsolver = rapper.StochSolver(None, tree_model=concrete_tree, fsfct=self.pysp_instance_creation_callback)
# ef_sol = stsolver.solve_ef('cplex', tee=True)
# print(ef_sol.solver.termination_condition)           --> optimal
#

# file = open('Output/Variable summary test.txt','w')  # file name has to have capital
#
# file.write('profit: {0}\n'.format(value(model.profit)))  # the second line is profit
# for v in model.component_objects(Var,active=True):
#     file.write("Variable %s\n" % v)  # \n makes new line
#     for index in v:
#         try:
#             if v[index].value > 0.0001:
#                 file.write("   %s %s\n" % (index,v[index].value))
#         except:
#             pass
# file.close()