##python modules
from dash import dcc, html, Input, Output, callback

##AFO modules
import RunAFO
import Functions as fun

##read in exp.xl
default_data, exp_group_bool = fun.f_read_exp()
inputs = default_data.iloc[0:2,:]
inputs.columns = inputs.iloc[0]
default_data = fun.f_group_exp(default_data, exp_group_bool)
exp_data = default_data.copy() #create a copy that can be updated

max_trials = 5
regions = ['GSM', 'CWM'] #these can just be entered here for now. These will populate the region dropdown menu.


##code to generate dropdown lists - this will be linked to callback when tabs are changed
tab_name='General' #todo this will be replaced by dash component
dropdown_list = list(inputs.loc[:,inputs.columns==tab_name])

##example code to access default values
input='wool price' #todo this would be replaced by dash input
input_bool = inputs == input
for trial_no in range(max_trials):
    default = exp_data.loc[trial_no,input_bool]

##example code to update exp_data
input='wool price' #todo this will be replaced by dash input
value = 50 #todo this will be replaced by dash input
input_bool = inputs == input
for trial_no in range(max_trials):
    exp_data.loc[trial_no,input_bool] = value



##example code to run AFO
n_trials=1 #todo this will be replaced by dash input
region='GSM' #todo this will be replaced by dash input
summary, lw = RunAFO.f_exp_app(exp_data, n_trials, region)



