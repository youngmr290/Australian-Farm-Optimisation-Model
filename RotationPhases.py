# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:00:30 2020

Version Control:
Version     Date        Person  Change
1.1         22/02/202   MRY      commented out con2 as it is not needed - don't delete incase we are wrong and it is required.

Known problems:
Fixed   Date    ID by   Problem


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
# rot_con2 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con2', header= None)
# rot_con2 = rot_con2.set_index([0,1])
# rot_con2 =rot_con2.squeeze().to_dict()



