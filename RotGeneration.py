# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:54:15 2020

This module generate the rotation info and writes it to excel. 
This is because it is a bit slow and only need to be recalculated\
when rotation rules are changed or something major

@author: young
"""
import numpy as np
import pandas as pd
import itertools
from openpyxl import load_workbook

#MIDAS modules
import Functions as fun
import UniversalInputs as uinp


yr0 = np.array(['b', 'h', 'o','of', 'w', 'f', 'l', 'z','r'
               , 'a', 'ar', 'a3', 'a4', 'a5'
               , 's', 'sr', 's3', 's4', 's5'
               , 'm', 'm3', 'm4', 'm5'
               , 'u', 'ur', 'u3', 'u4', 'u5'
               , 'x', 'xr', 'x3', 'x4', 'x5'
               , 'j', 't', 'jr', 'tr'])
yr1 = np.array(['AR', 'SR'
       ,'E', 'N', 'P', 'OF'
       , 'A', 'A3', 'A4', 'A5'
       , 'S', 'S3', 'S4', 'S5'
       , 'M', 'M3', 'M4', 'M5'
       , 'U', 'U3', 'U4', 'U5'
       , 'X', 'X3', 'X4', 'X5'
       , 'T', 'J'])
yr2 = np.array(['E', 'N', 'P'
       , 'A', 'A3', 'A4', 'A5'
       , 'S', 'S3', 'S4', 'S5'
       , 'M', 'M3', 'M4', 'M5'
       , 'U', 'U3', 'U4', 'U5'
       , 'X', 'X3', 'X4', 'X5'
       , 'T', 'J'])
yr3 = np.array(['E', 'N', 'P'
       , 'A', 'A3', 'A4', 'A5'
       , 'S', 'S3', 'S4', 'S5'
       , 'M', 'M3', 'M4', 'M5'
       , 'U', 'U3', 'U4', 'U5'
       , 'X', 'X3', 'X4', 'X5'
       , 'T', 'J'])
yr4 = np.array(['A','Y'])

arrays=[yr4,yr3,yr2,yr1,yr0]
phases=fun.cartesian_product_simple_transpose(arrays)


###########################################################
#params used to try make the rules more flexible to change#
###########################################################

##after how many yrs is annual resown
resow_a = 4
a_sow_col=np.size(phases,1)-1-resow_a #difference between history len and resow len -the col of the last phase yr that determines resseding (if 0 it means that all the history must be a crop for pasture to be resown)

'''
To understand this you must know the following;
true&false = false
true+false = true
true*true*false = false
'''


for i in range(np.size(phases,1)-1):
##drop rules 1; unprofitable
    ###no cont canola
    phases = phases[~(np.isin(phases[:,i], ['N'])&np.isin(phases[:,i+1], ['N','r','z']))] 
    ###no cont pulse
    phases = phases[~(np.isin(phases[:,i], ['P'])&np.isin(phases[:,i+1], ['P','l','f']))] 
    ###no pulse after pasture
    phases = phases[~(np.isin(phases[:,i], ['AR', 'SR','A','A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5','U','U3','U4','U5','T','T3','T4','T5'])&np.isin(phases[:,i+1], ['P','l','f']))] 
    ###no pasture after spraytoped 
    phases = phases[~(np.isin(phases[:,i], ['S','S3','S4','S5'])&np.isin(phases[:,i+1], ['AR', 'SR','A', 'A3', 'A4', 'A5','M','M3','M4','M5','S','S3','S4','S5','a','ar','a3','a4','a5','s','sr','s3','s4','s5','m','m3','m4','m5']))] 
    ###only spraytopped pasture after manipulated
    phases = phases[~(np.isin(phases[:,i], ['M','M3','M4','M5'])&np.isin(phases[:,i+1], ['AR','A', 'A3', 'A4', 'A5','M','M3','M4','M5','a','ar','a3','a4','a5','m','m3','m4','m5']))] 
    ###not going to resown tedera after a tedera (in a cont rotation you resow every 10yrs but that is accounted for with 'tc')
    phases = phases[~(np.isin(phases[:,i], ['T','J'])&np.isin(phases[:,i+1], ['tr','jr']))] 
    ###not going to resow lucerne after a lucerne (in a cont rotation you resow every 5yrs but that is accounted for with 'uc' & 'xc')
    phases = phases[~(np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['xr','ur']))] 
    ###cant have 1yr of perennial
    try: #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['T','J'])&~(np.isin(phases[:,i-1], ['T','J', 'Y']) + np.isin(phases[:,i+1], ['T','J','t','j'])))] 
    except IndexError: pass
    ###cant have 1yr of perennial
    try: #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['U','X'])&~(np.isin(phases[:,i-1], ['U','X', 'Y']) + np.isin(phases[:,i+1], ['U','X','u','x'])))] 
    except IndexError: pass
   
    ##drop rules 2; logical
    ###Annual 
    if i == 0:
        pass #A in yr 4 can be followed by any thing
    else:
        ###only A or A3 after A
        phases = phases[~(np.isin(phases[:,i], ['A'])&np.isin(phases[:,i+1], ['A4','A5','M4','M5','S4','S5','a4','a5','s4','s5','m4','m5','ar', 'sr','AR', 'SR']))] 
    ###pasture 4 must come after pasture 3    
    phases = phases[~(np.isin(phases[:,i], ['A3','M3'])&np.isin(phases[:,i+1], ['AR', 'SR','A', 'A3','A5','M','M3','M5','S','S3','S5','a','ar','a3','a5','s','sr','s3','s5','m','m3','m5']))] 
    ###pasture 5 must come after pasture 4
    phases = phases[~(np.isin(phases[:,i], ['A4','M4'])&np.isin(phases[:,i+1], ['AR', 'SR','A', 'A3','A4','M','M3','M4','S','S3','S4','a','ar','a3','a4','s','sr','s3','s4','m','m3','m4']))] 
    ###pasture 5 must come after pasture 5
    phases = phases[~(np.isin(phases[:,i], ['A5','M5'])&np.isin(phases[:,i+1], ['AR', 'SR','A', 'A3','A4','M','M3','M4','S','S3','S4','a','ar','a3','a4','s','sr','s3','s4','m','m3','m4']))] 
    ###cant have A3 after anything except A  (have used a double negitive here)
    phases = phases[~(~np.isin(phases[:,i], ['A'])&np.isin(phases[:,i+1], ['A3','S3','M3','a3','m3','s3']))] 
    ###cant have A3 after anything except A A (goes with the rule above)
    try: #used for conditions that are concerned with more than two yrs
        phases = phases[~(~np.isin(phases[:,i], ['A','AR', 'SR'])&np.isin(phases[:,i+2], ['A3','S3','M3','a3','m3','s3']))] 
    except IndexError: pass
    ###this if statement is required because in yr3 A4 and A5 can follow A
    if i ==0:
        ###cant have A4 after anything except A3  (have used a double negitive here)
        phases = phases[~(~np.isin(phases[:,i], ['A3','A'])&np.isin(phases[:,i+1], ['A4','S4','M4','a4','s4','m4']))] 
        ###cant have A5 after anything except A4  (have used a double negitive here)
        phases = phases[~(~np.isin(phases[:,i], ['A4','A'])&np.isin(phases[:,i+1], ['A5','M5','S5','a5','m5','s5']))] 
    else:
        ###cant have A4 after anything except A3  (have used a double negitive here)
        phases = phases[~(~np.isin(phases[:,i], ['A3'])&np.isin(phases[:,i+1], ['A4','S4','M4','a4','s4','m4']))] 
        ###cant have A5 after anything except A4  (have used a double negitive here)
        phases = phases[~(~np.isin(phases[:,i], ['A4','M4','M5','A5'])&np.isin(phases[:,i+1], ['A5','S5','M5','a5','s5','m5']))]
    ###can't have A after A A 
    try: #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['A'])&np.isin(phases[:,i+1], ['A','S','M'])&np.isin(phases[:,i+2], ['A','a','ar','M','m','S','s','sr']))] 
    except IndexError: pass
    
    ##Lucerne
    if i == 0:
        pass #U in yr 4 can be followed by anything
    else:
        ###only U or U3 ufter U
        phases = phases[~(np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['U4','U5','X4','X5','u4','u5','x4','x5']))] 
    phases = phases[~(np.isin(phases[:,i], ['U3','X3'])&np.isin(phases[:,i+1], ['U', 'U3','U5','X','X3','X5','u','ur','u3','u5','x','xr','x3','x5']))] #pasture 4 muxt come ufter pasture 3
    phases = phases[~(np.isin(phases[:,i], ['U4','X4'])&np.isin(phases[:,i+1], ['U', 'U3','U4','X','X3','X4','u','ur','u3','u4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 4
    phases = phases[~(np.isin(phases[:,i], ['U5','X5'])&np.isin(phases[:,i+1], ['U', 'U3','U4','X','X3','X4','u','ur','u3','u4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 5
    phases = phases[~(~np.isin(phases[:,i], ['U','X','Y'])&np.isin(phases[:,i+1], ['U3','X3','u3','x3']))] #cant have U3 after anything except U 
    try:  #used for conditions that are concerned with more than two yrs
        phases = phases[~(~np.isin(phases[:,i], ['U','X','Y'])&np.isin(phases[:,i+2], ['U3','X3','u3','x3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
    except IndexError: pass
    ###this if statement is required because in yr3 U4 and U5 can follow U
    if i == 0:
        phases = phases[~(~np.isin(phases[:,i], ['U3','X3','Y'])&np.isin(phases[:,i+1], ['U4','X4','u4','x4']))] #cant have U4 after anything except U3  
        phases = phases[~(~np.isin(phases[:,i], ['U4','X4','Y'])&np.isin(phases[:,i+1], ['U5','X5','u5','x5']))] #cant have U5 after anything except U4  
    else:    
        phases = phases[~(~np.isin(phases[:,i], ['U3','X3'])&np.isin(phases[:,i+1], ['U4','X4','u4','x4']))] #cant have U4 after anything except U3  
        phases = phases[~(~np.isin(phases[:,i], ['U4','X4','U5','X5'])&np.isin(phases[:,i+1], ['U5','X5','u5','x5']))] #cant have U5 after anything except U4 or U5
    ###can't have U ufter U U 
    try:  #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['U','X'])&np.isin(phases[:,i+2], ['U','X','u','x','ur','xr']))]
    except IndexError: pass

##lucerne and tedera resowing
phases = phases[~(~np.isin(phases[:,np.size(phases,1)-2], ['U','X'])&np.isin(phases[:,np.size(phases,1)-1], ['U','X','u','x']))] #lucerne after a non lucern must be resown
phases = phases[~(~np.isin(phases[:,np.size(phases,1)-2], ['T','J'])&np.isin(phases[:,np.size(phases,1)-1], ['T','J','t','j']))] #Tedera after a non tedera must be resown

##annual resowing
###if there is a previous annual then yr0 doesn't need to be resown
a_index =np.any(np.isin(phases[:,a_sow_col:np.size(phases,1)-1], ['AR', 'SR','A','A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5']), axis=1)&np.isin(phases[:,np.size(phases,1)-1], ['ar', 'sr'])
phases = phases[~a_index]
###if there is a previous annual then yr1 doesn't need to be resown
a_index1 =np.any(np.isin(phases[:,a_sow_col:np.size(phases,1)-2], ['A','A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5']), axis=1)&np.isin(phases[:,np.size(phases,1)-2], ['AR', 'SR'])
phases = phases[~a_index1]
###if there are not annuals in the history then an annual in yr0 must be resown
a_index2 = np.all(~np.isin(phases[:,a_sow_col:np.size(phases,1)-1], ['AR', 'SR','A','A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5']), axis=1)&np.isin(phases[:,np.size(phases,1)-1], ['a', 's','m'])
phases = phases[~a_index2]





##X can't be in the same rotation as U, T, J and A
a_xutj = np.any(np.isin(phases[:,:], ['AR', 'SR','ar','a','a3','a4','a5','A','A3','A4','A5','m','m3','m4','m5','M','M3','M4','M5','s','sr','s3','s4','s5','S','S3','S4','S5']), axis=1)&np.any(np.isin(phases[:,:], ['X','X3','X4','X5','x','xr','x3','x4','x5','U','U3','U4','U5','u','ur','u3','u4','u5','T','t','tr','J','j','jr']), axis=1)
x_utj = np.any(np.isin(phases[:,:], ['X','X3','X4','X5','x','xr','x3','x4','x5']), axis=1)&np.any(np.isin(phases[:,:], ['U','U3','U4','U5','u','ur','u3','u4','u5','T','t','tr','J','j','jr']), axis=1)
u_tj = np.any(np.isin(phases[:,:], ['U','U3','U4','U5','u','ur','u3','u4','u5']), axis=1)&np.any(np.isin(phases[:,:], ['T','t','tr','J','j','jr']), axis=1)
t_j = np.any(np.isin(phases[:,:], ['T','t','tr']), axis=1)&np.any(np.isin(phases[:,:], ['J','j','jr']), axis=1)
phases = phases[~(a_xutj + x_utj + u_tj + t_j)] 


######################################################
#delete cont rotations that require special handeling#
######################################################
'''
remove cont rotations before generilisation
-remove rotations that can provide them selves
-when generalising don't generalise yr1 
    - this will stop Y X X X3 x4 providing itself ie if you generalied yr1 to X then this rotation would provide itself'
'''
##check if every phase in a rotation is either lucerne or Y
xindex=np.all(np.isin(phases[:,:], ['Y','X5','x5','U5','u5']), axis=1) 
phases = phases[~xindex]
##check if every phase in a rotation is either T or Y
tindex=np.all(np.isin(phases[:,:], ['Y','T','J','t','j']), axis=1) 
phases = phases[~tindex]

##################
#continuous phase#
##################
tc=np.array(['tc','tc','tc','tc','tc'])
jc=np.array(['jc','jc','jc','jc','jc'])
uc=np.array(['uc','uc','uc','uc','uc'])
xc=np.array(['xc','xc','xc','xc','xc'])

phasess = np.concatenate((phases, [tc,jc,uc,xc])) #square brackets required because otherwise it thinks that the cont rotations are just 1D

################
#generalisation#
################
##yr 4 generilisation
g = np.size(phases,1)-1 #gets the number of the last col
### 'A' can be simplified to 'G' if there is another 'younger' pasture in the history and/or the current yr is not pasture (we only care about it for reseeding)
a_index = np.isin(phases[:,a_sow_col], ['A'])&(np.any(np.isin(phases[:,a_sow_col+1:g], ['AR', 'SR','A','A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5']), axis=1)+~np.isin(phases[:,g], ['a','ar','a3','a4','a5','s','sr','s3','s4','s5','m','m3','m4','m5']))
###use the created index to generalise
phases[a_index,a_sow_col]='G'
###'Y' can be simplified to 'G' if the current yr is not resown (we only need to distinguish between Y and G for phases where reseeding happens)
y_index = np.isin(phases[:,a_sow_col], ['Y'])&~np.isin(phases[:,g], ['ar','sr'])
###use the created index to generalise
phases[y_index,a_sow_col]='G'

##yr 2&3 generlisation
for i in range(2):
    gen = a_sow_col+1+i
    ###any annual can be generalised to 'A' if there is another younger annual 
    m_index = np.isin(phases[:,gen], ['A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5'])&np.any(np.isin(phases[:,gen+1:g+1], ['AR', 'SR','ar','a','a3','a4','a5','A','A3','A4','A5','m','m3','m4','m5','M','M3','M4','M5','s','sr','s3','s4','s5','S','S3','S4','S5']), axis=1)
    ###use the created index to generalise
    phases[m_index,gen]='A'
    ###a numbered lucerne can be generalised to 'U' if there is another younger lucerne 
    u_index = np.isin(phases[:,gen], ['U3','U4','U5'])&np.any(np.isin(phases[:,gen+1:g+1], ['U','U3','U4','U5','u','ur','u3','u4','u5']), axis=1)
    ###use the created index to generalise
    phases[u_index,gen]='U'
    ###a numbered lucerne can be generalised to 'X' if there is another younger lucerne 
    x_index = np.isin(phases[:,gen], ['X3','X4','X5'])&np.any(np.isin(phases[:,gen+1:g+1], ['X','X3','X4','X5','x','xr','x3','x4','x5']), axis=1)
    ###use the created index to generalise
    phases[x_index,gen]='X'

phases = np.unique(phases, axis=0)
   

##################
#history provide #
##################
hist_prov = phases[:,1:np.size(phases,1)].copy() #had to use copy so that the next steps didn't alter the phases array
##yr 4 generilisation
g = np.size(hist_prov,1)-1 #gets the number of the last col
### anything can be simplified to 'G' if there is another 'younger' pasture in the history, if not it can be simplified to A
a_index = np.any(np.isin(hist_prov[:,a_sow_col+1:g+1], ['AR', 'SR','ar','a','a3','a4','a5','A','A3','A4','A5','m','m3','m4','m5','M','M3','M4','M5','s','sr','s3','s4','s5','S','S3','S4','S5']), axis=1)
###use the created index to generalise
hist_prov[a_index,a_sow_col]='G'
###any annual in yr4 can be generalised to A
a2_index = np.isin(hist_prov[:,a_sow_col], ['A3','A4','A5','M3','M4','M5','S3','S4','S5'])
###use the created index to generalise
hist_prov[a2_index,a_sow_col]='A'
###any left now that is not an 'A' or 'G' in yr4 can be generalised to Y
ag_index = ~np.isin(hist_prov[:,a_sow_col], ['A','G'])
###use the created index to generalise
hist_prov[ag_index,a_sow_col]='Y'



##yr 2&3 generlisation
for i in range(2):
    gen = a_sow_col+1+i
    ###any annual can be generalised to 'A' if there is another younger annual 
    m_index = np.isin(hist_prov[:,gen], ['A3','A4','A5','M','M3','M4','M5','S','S3','S4','S5'])&np.any(np.isin(hist_prov[:,gen+1:g+1], ['ar','a','a3','a4','a5','A','AR','A3','A4','A5','m','m3','m4','m5','M','M3','M4','M5','s','sr','s3','s4','s5','S','SR','S3','S4','S5']), axis=1)
    ###use the created index to generalise
    hist_prov[m_index,gen]='A'
    ###a numbered lucerne can be generalised to 'U' if there is another younger lucerne 
    u_index = np.isin(hist_prov[:,gen], ['U3','U4','U5'])&np.any(np.isin(hist_prov[:,gen+1:g+1], ['U','U3','U4','U5','u','ur','u3','u4','u5']), axis=1)
    ###use the created index to generalise
    hist_prov[u_index,gen]='U'
    ###a numbered lucerne can be generalised to 'X' if there is another younger lucerne 
    x_index = np.isin(hist_prov[:,gen], ['X3','X4','X5'])&np.any(np.isin(hist_prov[:,gen+1:g+1], ['X','X3','X4','X5','x','xr','x3','x4','x5']), axis=1)
    ###use the created index to generalise
    hist_prov[x_index,gen]='X'

##change OF, SR and AR back to general for yr2 (these were specilised in yr1)
of_index = np.isin(hist_prov[:,g-1], ['OF'])
###use the created index to generalise
hist_prov[of_index,gen]='E'
##change OF, SR and AR back to general for yr2 (these were specilised in yr1)
sr_index = np.isin(hist_prov[:,g-1], ['SR'])
###use the created index to generalise
hist_prov[sr_index,gen]='S'
##change OF, SR and AR back to general for yr2 (these were specilised in yr1)
ar_index = np.isin(hist_prov[:,g-1], ['AR'])
###use the created index to generalise
hist_prov[ar_index,gen]='A'

###yr1 have to generalise from the specific phase to the phase set that is important in yr1 ie b goes to E
###generalise cereals
e_index = np.isin(hist_prov[:,g], ['b', 'h', 'o', 'w']) #of is not in this because we need to know if yr 1 is of because increases pasture germ due to less weed control
###use the created index to generalise
hist_prov[e_index,g]='E'
###generalise fodder
of_index = np.isin(hist_prov[:,g], ['of']) 
###use the created index to generalise
hist_prov[of_index,g]='OF'
###generalise pulse
p_index = np.isin(hist_prov[:,g], ['f','i', 'k', 'l', 'v']) 
###use the created index to generalise
hist_prov[p_index,g]='P'
###generalise tedera
p_index = np.isin(hist_prov[:,g], ['t','tr']) 
###use the created index to generalise
hist_prov[p_index,g]='T'
###generalise tedera manip
p_index = np.isin(hist_prov[:,g], ['j', 'jr']) 
###use the created index to generalise
hist_prov[p_index,g]='J'
###generalise resown lucern - the rest of lucerne just needs to be capitilised
p_index = np.isin(hist_prov[:,g], ['ur']) 
###use the created index to generalise
hist_prov[p_index,g]='U'
###generalise resown lucern - the rest of lucerne just needs to be capitilised
p_index = np.isin(hist_prov[:,g], ['xr']) 
###use the created index to generalise
hist_prov[p_index,g]='X'
###generalise canola
n_index = np.isin(hist_prov[:,g], ['z','r']) 
###use the created index to generalise
hist_prov[n_index,g]='N'
###convert all to capital (this will change a5 to A5 ect)
hist_prov = np.char.upper(hist_prov)
###create a copy of the generalised rotation provide - this is used in the section below when building con2 because i need a copy of the rot_provide for each phase ie before duplicates are removed
rot_hist_prov = hist_prov
###drop duplicates
hist_prov = np.unique(hist_prov, axis=0)

##################
#history require #
##################
hist_req = phases[:,0:np.size(phases,1)-1]
hist_req = np.unique(hist_req, axis=0)





############################################################################################################################################################################################
############################################################################################################################################################################################
#Generate the paramater for rotation provide and require
############################################################################################################################################################################################
############################################################################################################################################################################################


tuples_phases = [tuple(x) for x in phases]
tuples_hist_req = [tuple(x) for x in hist_req]
tuples_hist_prov = [tuple(x) for x in hist_prov]

l_phases = [''.join(x) for x in phases.astype(str)]
l_hist_req = [''.join(x) for x in hist_req.astype(str)]
l_hist_prov = [''.join(x) for x in hist_prov.astype(str)]


##################
#con 1 param     #
##################
'''determines what history1 each rotation requires and provides'''
mps_bool=[]
for rot_phase in phases:
    test=0
    test2=0
    for hist in hist_req:
        rot_phase_req=[]
        rot_phase_prov=[]
        l_hist=[]
        for i in range(len(hist)):
            ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set) so the hist must be altered in yr1
            if i == 3 and hist[i] == 'A':
                hist_set = {'a'}#uinp.structure['A1']
            elif i == 3 and hist[i] == 'S':
                hist_set = {'s'}
            elif i == 3 and hist[i] == 'M':
                hist_set = {'m'}
            elif i == 3 and hist[i] == 'E':
                hist_set = uinp.structure['E1']
            else: hist_set = uinp.structure[hist[i]]
            # hist_set = uinp.structure[hist[i]]
            l_hist.append(hist_set) #deterimines the sets in each constraint
            
            ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set) so the hist must be altered in yr1
            if i == 3 and rot_phase[i] == 'A':
                req_set = {'a'}#uinp.structure['A1']
            elif i == 3 and rot_phase[i] == 'S':
                req_set = {'s'}
            elif i == 3 and rot_phase[i] == 'M':
                req_set = {'m'}
            elif i == 3 and rot_phase[i] == 'E':
                req_set = uinp.structure['E1']
            else: req_set = uinp.structure[rot_phase[i]]            
            rot_phase_req.append(req_set) #appends each set that corresponds to the letters in the rot_phase (required)
            rot_phase_prov.append(uinp.structure[rot_phase[i+1]]) #appends each set that corresponds to the letters in the rot_phase (provides)
        req=1
        prov=-1
        for i in range(len(hist)):
            req*=l_hist[i].issuperset(rot_phase_req[i]) #checks each set in a given rotation for the req part of the equation
            prov*=l_hist[i].issuperset(rot_phase_prov[i]) #checks each set in a given rotation for the prov part of the equation
        test+=prov
        test2+=req
        mps_bool.append(req+prov)
    if test==0: #doesn't provide a history
        print('rot doesnt provide a history: ',rot_phase)
    if test2==0: #doesn't provide a history
        print('rot doesnt req a history: ',rot_phase)
mps_bool=pd.Series(mps_bool) #convert to series because easier to manipulate
rot_phase_by_constrain = pd.DataFrame(list(itertools.product(l_phases,l_hist_req) ) ) #had to use this cartesian method as i couldn't get the fast function to work
mps_bool=pd.concat([rot_phase_by_constrain, mps_bool], axis=1) #add two dfs together 
mps_bool = mps_bool[(mps_bool.iloc[:,2] != 0)]

# rot_phase_by_constrain=rot_phase_by_constrain.set_index([0,1])
# rot_phase_by_constrain=rot_phase_by_constrain.iloc[:,0]+rot_phase_by_constrain.iloc[:,1]
# mps_bool.index=(rot_phase_by_constrain.index) #add index (constraint name) to mps value 

##################
#con 2 param     #
##################

'''
Determines what rotation each history2 requires and provides
con2 is more specific than con1.
This ensures that each rotation is used to provide another rotation

The sets are altered for yr1 - this is done using if statements which make the sets more specific
'''
mps_bool2=[]
for rot_phase, rot_prov in zip(phases,rot_hist_prov):
    test=0
    test2=0
    for hist in hist_prov:
        rot_phase_req=[]
        rot_phase_prov=[]
        l_hist=[]
        for i in range(len(hist)):
            ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set)
            if  rot_prov[i] == 'A':
                prov_set = {'a'}#uinp.structure['A1']
            elif  rot_prov[i] == 'S':
                prov_set = {'s'}
            elif  rot_prov[i] == 'M':
                prov_set = {'m'}
            elif  rot_prov[i] == 'E':
                prov_set = uinp.structure['E1']
            else: prov_set = uinp.structure[rot_prov[i]]
            rot_phase_prov.append(prov_set) #appends each set that corresponds to the letters in the rot_phase (required)
            # ##appends each set that corresponds to the letters in the rot_phase
            # rot_phase_prov.append(uinp.structure[rot_prov[i]]) 
            ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set)
            if rot_phase[i] == 'A':
                req_set = {'a'}#uinp.structure['A1']
            elif  rot_phase[i] == 'S':
                req_set = {'s'}
            elif  rot_phase[i] == 'M':
                req_set = {'m'}
            elif rot_phase[i] == 'E':
                req_set = uinp.structure['E1']
            else: req_set = uinp.structure[rot_phase[i]]
            rot_phase_req.append(req_set) #appends each set that corresponds to the letters in the rot_phase (required)
            ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set) so the hist must be altered in yr1
            if  hist[i] == 'A':
                hist_set = {'a'}#uinp.structure['A1']
            elif  hist[i] == 'S':
                hist_set = {'s'}
            elif  hist[i] == 'M':
                hist_set = {'m'}
            # if i == 3 and hist[i] == 'A':
            # hist_set = uinp.structure['A1']
            elif  hist[i] == 'E':
                hist_set = uinp.structure['E1']
            else: hist_set = uinp.structure[hist[i]]
            l_hist.append(hist_set) #deterimines the sets in each constraint
        prov=-1
        req=1
        for i in range(len(hist)):
            req*=rot_phase_prov[i].issuperset(l_hist[i]) #checks each set in a given rotation for the req part of the equation
            prov*=rot_phase_req[i].issuperset(l_hist[i]) #checks each set in a given rotation for the prov part of the equation
        test+=prov
        test2+=req
        mps_bool2.append(req+prov)
    if test==0: #doesn't provide a history
        print('doesnt require a con2: ',rot_phase)
    if test2==0: #doesn't provide a history
        print('rot doesnt req a history: ',rot_phase)
mps_bool2=pd.Series(mps_bool2) #convert to series because easier to manipulate
rot_phase_by_constrain2 = pd.DataFrame(list(itertools.product(l_phases,l_hist_prov) ) ) #had to use this cartesian method as i couldn't get the fast function to work
mps_bool2=pd.concat([rot_phase_by_constrain2, mps_bool2], axis=1) #add two dfs together 
mps_bool2 = mps_bool2[(mps_bool2.iloc[:,2] != 0)]


# mps_bool2=pd.Series(mps_bool2) #convert to series because easier to manipulate
# rot_phase_by_constrain2 = pd.DataFrame(list(itertools.product(tuples_phases,tuples_hist_prov) ) ) #had to use this cartesian method as i couldn't get the fast function to work
# rot_phase_by_constrain2=rot_phase_by_constrain2.iloc[:,0]+rot_phase_by_constrain2.iloc[:,1]
# mps_bool2.index=(rot_phase_by_constrain2) #add index (constraint name) to mps value 
# mps_bool2 = mps_bool2[(mps_bool2 != 0)]
# mps_bool2=[]
# for rot_phase, rot_prov in zip(phases,rot_hist_prov):
#     test=0
#     for hist in hist_prov:
#         rot_phase_req=[]
#         rot_phase_prov=[]
#         l_hist=[]
#         for i in range(len(hist)):
#             ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set)
#             if i == 3 and rot_prov[i] == 'A':
#                 prov_set = {'a'}#uinp.structure['A1']
#             elif i == 3 and rot_prov[i] == 'S':
#                 prov_set = {'s'}
#             elif i == 3 and rot_prov[i] == 'M':
#                 prov_set = {'m'}
#             elif i == 3 and rot_prov[i] == 'E':
#                 prov_set = uinp.structure['E1']
#             else: prov_set = uinp.structure[rot_prov[i]]
#             rot_phase_prov.append(prov_set) #appends each set that corresponds to the letters in the rot_phase (required)
#             # ##appends each set that corresponds to the letters in the rot_phase
#             # rot_phase_prov.append(uinp.structure[rot_prov[i]]) 
#             ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set)
#             if i == 3 and rot_phase[i] == 'A':
#                 req_set = {'a'}#uinp.structure['A1']
#             elif i == 3 and rot_phase[i] == 'S':
#                 req_set = {'s'}
#             elif i == 3 and rot_phase[i] == 'M':
#                 req_set = {'m'}
#             elif i == 3 and rot_phase[i] == 'E':
#                 req_set = uinp.structure['E1']
#             else: req_set = uinp.structure[rot_phase[i]]
#             rot_phase_req.append(req_set) #appends each set that corresponds to the letters in the rot_phase (required)
#             ##in yr 1 we don't want ar to provide a ie we don't want ar to be in the a set (same for fodder of in the E set) so the hist must be altered in yr1
#             if i == 3 and hist[i] == 'A':
#                 hist_set = {'a'}#uinp.structure['A1']
#             elif i == 3 and hist[i] == 'S':
#                 hist_set = {'s'}
#             elif i == 3 and hist[i] == 'M':
#                 hist_set = {'m'}
#             # if i == 3 and hist[i] == 'A':
#             # hist_set = uinp.structure['A1']
#             elif i == 3 and hist[i] == 'E':
#                 hist_set = uinp.structure['E1']
#             else: hist_set = uinp.structure[hist[i]]
#             l_hist.append(hist_set) #deterimines the sets in each constraint
#         prov=-1
#         req=1
#         for i in range(len(hist)):
#             req*=rot_phase_prov[i].issuperset(l_hist[i]) #checks each set in a given rotation for the req part of the equation
#             prov*=rot_phase_req[i].issuperset(l_hist[i]) #checks each set in a given rotation for the prov part of the equation
#         test+=prov
#         mps_bool2.append(req+prov)
#     if test==0: #doesn't provide a history
#         print('doesnt require a con2: ',rot_phase)
# mps_bool2=pd.Series(mps_bool2) #convert to series because easier to manipulate
# rot_phase_by_constrain2 = pd.DataFrame(list(itertools.product(tuples_phases,tuples_hist_prov) ) ) #had to use this cartesian method as i couldn't get the fast function to work
# rot_phase_by_constrain2=rot_phase_by_constrain2.iloc[:,0]+rot_phase_by_constrain2.iloc[:,1]
# mps_bool2.index=(rot_phase_by_constrain2) #add index (constraint name) to mps value 
# mps_bool2 = mps_bool2[(mps_bool2 != 0)]#.to_dict()


############################################################################################################################################################################################
############################################################################################################################################################################################
#Write rotations and rotation provide/require stuff to excel
############################################################################################################################################################################################
############################################################################################################################################################################################

##start writing
writer = pd.ExcelWriter('Rotation.xlsx', engine='xlsxwriter')
##list of rotations - index: tupple, values: expanded version of rotation
rot_phases =  pd.DataFrame(phases, index=l_phases)
rot_phases.to_excel(writer, sheet_name='rotation list',index=True,header=False)
##con1 - the paramater for which history each rotation provides and requires
mps_bool.to_excel(writer, sheet_name='rotation con1',index=False,header=False)
##con1 set - passed into the pyomo constraint
hist_req = pd.DataFrame(hist_req, index=l_hist_req)
hist_req.to_excel(writer, sheet_name='rotation con1 set',index=True,header=False)
##con2
mps_bool2.to_excel(writer, sheet_name='rotation con2',index=False,header=False)
##con2 set - passed into the pyomo constraint
hist_prov = pd.DataFrame(hist_prov, index=l_hist_prov)
hist_prov.to_excel(writer, sheet_name='rotation con2 set',index=True,header=False)


##finish writing and save
writer.save()




























