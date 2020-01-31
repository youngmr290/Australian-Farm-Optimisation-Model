# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:00:30 2020

@author: young
"""

#python modules
import pandas as pd
import numpy as np
import timeit


#MUDAS modules
import UniversalInputs as uinp
import Functions as fun


###########################
#rotation phase low bound #
###########################
##debuging rotaion constraints- create dict for each rotation with 0 as default, the loop through changing each value to 1 (this dict will be used as the min bound)
phases_df = uinp.structure['phases']
# phase_ind=uinp.structure['phases'].set_index(list(range(uinp.structure['phase_len']))).index
lo_bound = dict.fromkeys(phases_df.index, 0)#create default dict

#############################
#rotation phase constraint1 #
#############################    
rot_con1 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con1', header= None)#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
rot_con1 = rot_con1.set_index([0,1])
rot_con1 =rot_con1.squeeze().to_dict()
rot_con2 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con2', header= None)
rot_con2 = rot_con2.set_index([0,1])
rot_con2 =rot_con2.squeeze().to_dict()



# def rot_phase_mps():
#     '''determines what history1 each rotation requires and provides'''
#     rot_phases=pd.Series(uinp.structure['rotations']['rot_phase']).dropna() #convert to series then drop nan
#     constraints=pd.Series(uinp.structure['rotations']['constraints']).dropna() #convert to series then drop nan
#     mps_bool=[]
#     for rot_phase in rot_phases:
#         test=0
#         for hist in constraints:
#             rot_phase_split=rot_phase.split()
#             rot_phase_split.reverse()
#             hist_split=hist.split()
#             hist_split.reverse()
#             rot_phase_req=[]
#             rot_phase_prov=[]
#             for i in range(len(hist_split)):
#                 rot_phase_req.append(uinp.structure[rot_phase_split[i+1]]) #appends each set that corresponds to the letters in the rot_phase (required)
#                 rot_phase_prov.append(uinp.structure[rot_phase_split[i]]) #appends each set that corresponds to the letters in the rot_phase (provides)
#                 hist_split[i]=uinp.structure[hist_split[i]] #deterimines the sets in each constraint
#             req=1
#             prov=-1
#             for i in range(len(hist_split)):
#                 req*=hist_split[i].issuperset(rot_phase_req[i]) #checks each set in a given rotation for the req part of the equation
#                 prov*=hist_split[i].issuperset(rot_phase_prov[i]) #checks each set in a given rotation for the prov part of the equation
#             test+=prov
#             mps_bool.append(req+prov)
#         # if test==0:
#             # print(rot_phase)
#     mps_bool=pd.Series(mps_bool) #convert to series because easier to manipulate
#     rot_phase_by_constrain=pd.DataFrame(fun.cartesian_product_simple_transpose([rot_phases,constraints]))#.astype(str) #determines all possible combinations of constraint and rotation phase
#     rot_phase_by_constrain=rot_phase_by_constrain[0]+' '+rot_phase_by_constrain[1] #combine two cols into one so the string can be split
#     rot_phase_by_constrain=rot_phase_by_constrain.str.split(expand=True) #splits strings into individual cells
#     mps_bool.index=( rot_phase_by_constrain) #add index (constraint name) to mps value 
#     return mps_bool[(mps_bool != 0)].to_dict()

# def rot_phase_mps2():
#     '''
#     Determines what rotation each history2 requires and provides
#     History2 is more specific than history1.
#     This ensures that each rotation is used to provide another rotation
#     '''
#     rot_phases=pd.Series(uinp.structure['rotations']['rot_phase']).dropna() #convert to series then drop nan
#     constraints2=pd.Series(uinp.structure['rotations']['constraints2']).dropna() #convert to series then drop nan
#     mps_bool=[]
#     for rot_phase in rot_phases:
#         for hist in constraints2:
#             rot_phase_split=rot_phase.split()
#             rot_phase_split.reverse()
#             hist_split=hist.split()
#             hist_split.reverse()
#             rot_phase_req=[]
#             rot_phase_prov=[]
#             for i in range(len(hist_split)):
#                 rot_phase_req.append(uinp.structure[rot_phase_split[i+1]]) #appends each set that corresponds to the letters in the rot_phase (required)
#                 rot_phase_prov.append(uinp.structure[rot_phase_split[i]]) #appends each set that corresponds to the letters in the rot_phase (provides)
#                 hist_split[i]=uinp.structure[hist_split[i]] #deterimines the sets in each constraint
#             prov=1
#             req=-1
#             for i in range(len(hist_split)):
#                 req*=rot_phase_req[i].issuperset(hist_split[i]) #checks each set in a given rotation for the req part of the equation
#                 prov*=rot_phase_prov[i].issuperset(hist_split[i]) #checks each set in a given rotation for the prov part of the equation
#             mps_bool.append(req+prov)
#     mps_bool=pd.Series(mps_bool) #convert to series because easier to manipulate
#     rot_phase_by_constrain=pd.DataFrame(fun.cartesian_product_simple_transpose([rot_phases,constraints2]))#.astype(str) #determines all possible combinations of constraint and rotation phase
#     rot_phase_by_constrain=rot_phase_by_constrain[0]+' '+rot_phase_by_constrain[1] #combine two cols into one so the string can be split
#     rot_phase_by_constrain=rot_phase_by_constrain.str.split(expand=True) #splits strings into individual cells
#     mps_bool.index=( rot_phase_by_constrain) #add index (constraint name) to mps value 
#     return mps_bool[(mps_bool != 0)].to_dict()



