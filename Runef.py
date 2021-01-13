'''
Module used to construct lp file from discrete model and stocastic add ons.
'''



import subprocess
from pyomo.core import *

from ReferenceModel import model

subprocess.call(['C:/Users/21512438/Anaconda3/Scripts/runef.exe', '--traceback', '--symbolic-solver-labels',
                 '--verbose', '-mReferenceModel.py', '--solve', '--solver=glpk'])

#^to test - Thomas for forum
def pysp_scenario_tree_model_callback():
    # Return a NetworkX scenario tree.
     g = networkx.DiGraph()

    ce1 = "Stage_0"
    g.add_node("Root",
           cost=ce1,
           variables=var_lists[0],
           derived_variables=[])

    ce2 = "Stage_1"
    g.add_node("LowPriceScenario",
           cost=ce2,
           variables=var_lists[1],
           derived_variables=[])
    g.add_edge("Root", "LowPriceScenario", weight=0.999)

    g.add_node("HighPriceScenario",
           cost=ce2,
           variables=var_lists[1],
           derived_variables=[])
    g.add_edge("Root", "HighPriceScenario", weight=0.001)

    return g
# the pysp_instance_creation_callback looks like this:

def pysp_instance_creation_callback(self, scenario_name, node_names):

    model = self.model
    scenario_data = self.scenario_data

    # create instances for different szenarios and write the unique scenariodata into the instance
    instance = model.clone()   #todo solution time beschleunigen
    instance.stochastic_data.store_values(scenario_data[scenario_name])
    return instance

def addStageCost(self,stages_dict,component_dict,model):
    for stage in stages_dict.keys():
        if stage == 0:
            T = range(stages_dict[stage])

        else:
            T = range(stages_dict[stage - 1],stages_dict[stage])

        model.add_component('Stage_' + str(stage),Expression(expr=cost1 + cost2)))

        return model

    # then I execute the following code:

model = self.addStageCost(stages_dict=stages_dict,component_dict=component_dict,model=model)


concrete_tree = self.pysp_scenario_tree_model_callback(stages_dict=stages_dict, component_dict=component_dict, model=model)
stsolver = rapper.StochSolver(None, tree_model=concrete_tree, fsfct=self.pysp_instance_creation_callback)
ef_sol = stsolver.solve_ef('cplex', tee=True)
print(ef_sol.solver.termination_condition)           --> optimal


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