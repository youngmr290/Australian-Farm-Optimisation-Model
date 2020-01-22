# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:39:41 2020

@author: young
"""
#################################################################################################################################################################
#################################################################################################################################################################
#actual code
#################################################################################################################################################################
#################################################################################################################################################################

###########################
#alternative code methods #
###########################

# 1a - disagregated version of rotation constraint, was used to help debug
def rot_phase_link(model,l,h1,h2,h3,h4):
    print('history :',((h1,)+(h2,)+(h3,)+(h4,)))
    test=0
    for r in model.phases:
        index=((r)+(h1,)+(h2,)+(h3,)+(h4,))
        #print('initial: ',index)
        if index in model.rot_phase_link:
            print('success: ',index)
            test+=model.num_phase[r,l]*model.rot_phase_link[r,h1,h2,h3,h4]
            # print('success to follow')
    print(test)
    return  test<=0
model.jj = Constraint(model.lmus,model.rot_constraints, rule=rot_phase_link, doc='rotation phases constraint')





############################################
#previous code methods - depreciated code  #
############################################

#origional method for determining phases df
phases_df = fun.phases(inp.input_data['phases'],inp.input_data['phase_len'])




#################################################################################################################################################################
#################################################################################################################################################################
# general coding info 
#################################################################################################################################################################
#################################################################################################################################################################

############################
#df                        #
############################

'''
Change index - this is useful when you want to preform calcs with multiple dfs
    - reindex, this is easy when there is one common level incommon, can also be done when nothing incommon but often results in nan
    -pd.concat, keys - can be used to add level to df (ie in stubble module; pd.concat([vol]*len(inp.input_data['sheep_pools']), keys=inp.input_data['sheep_pools']))
    -df.columns = pd.MultiIndex.from_product([df.columns, ['C']]) this adds c to the index, so you would then have to use reindex to apply (ie grain price in crop module)

Df to Series
    -.iloc[:,0]
    -.squeeze()



#delete col with no header #
    this is here becayse dropping a col with no header is not compatible with the usual way
    this if statement checks if the column has a header and then deletes the nessecary way
        if list(df.columns)[0]:
            df=df.drop(df.columns[0],axis=1)
'''

############################
#dicts                     #
############################
''' 
df to dict;
    - default; colunm names become keys. but colums of the df must be selected to stop it becoming a nested dict (using .to_dict())
    - can transpose to get index as key and cols as values eg:
        mmmm = df.transpose()['Col_1'].to_dict() #uses index as key and col as values
    - however you if you stack the df so that it is 1D then the index (row) will become the keys (remember to set the index before stacking)
    - you can also select two columns one with the keys one with the value using the dict(args) function
            eg if lakes is your DataFrame, you can do something like: 
                area_dict = dict(zip(lakes.area, lakes.count)) #takes two cols, the first becomes key second value
'''

############################
#time series               #
############################

''' 
converrt to int number of days
    - int.days() 

simple time conversion (python basically guesses what time you mean from inputs) -good for simple dates ie 1-1-19
    - date = parse(key, dayfirst = True)   
for more complex ones you can use sharfttime or something like that

Convert dt to int
    -To convert to exact days ie 1day 12hours would convert to 1.5 (D can be changed to M,Y,s for other conversions)
        -date / np.timedelta64(1, 'D')
    - Another method to convert however this doesn't do decimals ie 1day 12hours would convert to 1. To get around this you can convert to seconds then divide by 86400 (60*60*24) to get days as int 
        -.astype('timedelta64[s]') - can use D to get days
    
Access an attribute (ie get the number of days in a dt)
    - Use the dt.days attribute. Access this attribute via: (note if it is 1day 0 seconds, dt.seconds will return 0 not 3600)
        -.dt.days (You can also get the seconds and microseconds attributes in the same way)
'''

############################
#pyomo                     #
############################
'''
Pyomo - if there is a complex variable bound ie one that may be different for different variables within a set such as casual labour which may
        have a different bound for each labour period, you can use constraint to bound it because the variable bounds are not flexible enough
        
delete blocks - sometimes the index also needs to be deleted (not sure what that really means or why it is required)
    -model.del_component(model.num_phase_index)
    -model.del_component(model.num_phase)


'''

'''

############################
#time code                 #
############################
'''
import timeit
print(timeit.timeit(test,number=10)/10)
'''

#################################################################################################################################################################
#################################################################################################################################################################
## possible causes of error#
#################################################################################################################################################################
#################################################################################################################################################################

#casual and perm staff now require farmer to supervise, this could be a somethig that makes the model behave strange (casual staff could become limited due to lack of supervision time).
