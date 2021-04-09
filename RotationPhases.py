# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:00:30 2020

Version Control:
Version     Date        Person  Change
1.1         22/02/202   MRY      commented out con2 as it is not needed - don't delete in case we are wrong and it is required.

Known problems:
Fixed   Date    ID by   Problem


@author: young
"""

#python modules
import pandas as pd
import numpy as np
import timeit


#AFO modules
import StructuralInputs as sinp
import PropertyInputs as pinp

def landuses_phases(params,report):
    '''function to store phases into report dictionary'''
    phases=sinp.phases['phases']
    phases_rk = phases.set_index(5, append=True) #add landuse as index level
    params['phases_rk'] = dict.fromkeys(phases_rk.index,1)
    report['phases']=sinp.phases['phases']
    report['all_pastures']=sinp.landuse['All_pas'] #all_pas2 includes the cont pasture landuses


def rot_params(params):
    ##area
    params['lmu_area'] =  pinp.general['lmu_area'].squeeze().to_dict()

    #############################
    #rotation phase constraint1 #
    #############################    
    rot_req = pd.read_excel('Rotation.xlsx', sheet_name='rotation_req', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_prov = pd.read_excel('Rotation.xlsx', sheet_name='rotation_prov', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_req = rot_req.set_index([0,1])
    rot_prov = rot_prov.set_index([0,1])
    # params['rot_req_keys'] = rot_req.squeeze().to_dict().keys()
    params['hist_prov'] = rot_prov.squeeze().to_dict()
    params['hist_req'] = rot_req.squeeze().to_dict()



