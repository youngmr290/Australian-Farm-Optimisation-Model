"""
author: young

.. _RotGeneration module:

Background
----------

This module generates the rotation phases and writes them to excel.
This is because it is a bit slow and only needs to be recalculated
when rotation rules are changed or new land uses are added.

There are three options
   1. Generate a large list of rotations using RotGeneration.py. The rotation generator generates all rotation options
      as default. This results in slow model execution time due to relatively long Pyomo matrix construction time.
      However, the user has the ability to remove certain landuses before generating the array of phases (e.g. tedera
      and lucerne can be removed). Simulation modelling is used to generate the inputs.
   2. Generate a ‘custom’ list of rotations using RotGeneration.py. The user can specify the rotations they want
      generated by populating the ds_user_rot_init array and, at the top of the module, turn the boolean setting
      called customised_rotations to True and re-run the module. Simulation modelling is used to generate the inputs for these rotations.
   3. Generate a custom list of rotations and their inputs. The user can input the phases and their respective
      yield, chem and fert requirements into property.xlsx. RotGeneration.py is still required to be executed because
      it generates the coefficients for each rotation activity for the rotation constraints.


Automatic phase generation
--------------------------
The process for generating the phases was developed in conjunction with WA cropping advisors and is outlined below:

    #. Generate all combinations of the landuses that are possible in each year. The necessary landuses to represent reduces for the older years in the history because the rotations can be generalised.
    #. Run the ‘drop’ rules to delete those phases that are not required because they are unprofitable or illogical and don’t need to be included.
    #. Create the ‘history required’ = Yr5, Yr4, Yr3, Yr2, Yr1
    #. Create the ‘history provided’ = Yr4, Yr3, Yr2, Yr1, Yr 0

Parameters that define the rotation phases.

    These are the fundamental concepts that control the generalisation of rotation phases. If multiple land uses in the
    history are deemed to have the same effect on the current land use then they can be clustered which reduces the
    number of rotation phases which need to be represented. For example, if the assumption is that all cereal crops
    three years ago have the same impact on the current land use then we can reduce the number of rotations phases
    by clustering all cereal crop in yr3 of the history. See below for the current rules used for clustering.

    .. note:: rotations are built in an inflexible way currently, in the future it would be good to build them more
        automatically using the params below.

    Number of years:

    * of consecutive cereal before yields plateau (c_cereal = 3)
    * of consecutive non-annual pasture before annual pasture needs to be resown (resow_a = 4)
    * of consecutive annual pasture before annual pasture needs to be spraytopped (spray_a = 4). This is required because overtime the pasture composition becomes less ideal (ie brome grass)
    * ...ditto for lucerne (resow_u = 1)
    * ...ditto for tedera (resow_t = 1)
    * that spraytopping or pasture manipulation impacts on crop yields or pasture seed bank (spraytop = 2). Meaning
      that two spraytops has greater weed control than one.
    * that pasture manipulation followed by spraytopping impacts on yields or seed bank (manip_sprayt=2) on the assumption that manipulation + spraytopping has a greater effect than spraytopping alone.
    * of legume pasture that will still be increasing soil N levels (build_n = 5)
    * of non-legume crop to utilise all soil organic N, if built to maximum levels (use_n = 3)
    * of annual pasture till the seed bank is fully replenished and independent of crop history (build_seed = 1)
    * of consecutive non-annual that continues to deplete the seed bank (deplete_seed = 3)
    * of non-canola till canola disease levels have decreased to a minimum (canola_disease = 3)
    * of non-pulse till pulse disease levels have decreased to a minimum (pulse_disease = 3)
    * of years of continuous tedera before it needs resowing (tedera_resowing = 10)
    * of Lucerne before it needs to be resown (lucerne_resowing = 5)

    Note: disease can effects crops in the same class quite differently (e.g. barley is often a reasonable break crop
    for wheat). Because of this crops can not be grouped until yr3 when disease reaches a plateau.

    Length of rotation phase = maximum(resow_a, resow_u, resow_t, build_n, use_n, deplete_seed, canola disease,
    pulse disease, spraytop, manip_sprayt) + 1

The following rules are implemented to remove illogical rotation phases:

    #. If it is Lucerne after non-Lucerne then it must be resown

    #. If it is Tedera after non-Tedera then it must be resown

    #. If it is annual pasture after 4 other non-annual pasture phases then the annual pasture must be resown

    #. If it is annual pasture after 4 other annual pasture phases then the annual pasture must be spraytopped

The following rules are implemented to remove unprofitable rotation phases:

    .. note:: These rules may need to be tweaked for certain analyses. For example, in the seasonal variation model it may
        be sensible to include a after s and m or maybe s after s and m after m. This will allow tactical adjustment to
        remain with annual pasture even though cropping was the strategic decision.

    #. No continuous canola (due to canola disease)

    #. Two years between pulse crops (due to pulse disease)

    #. No pulse crop after a pasture (this would constitute a poor use of N fixed by the legume pasture)

    #. No annual pasture (other than spray-topped pasture) after a manipulated annual pasture (usually the
       purpose of pasture manipulation is to help prepare a field for subsequent cropping. However, cropping
       aside, a spray-topped pasture is feasible use of a field that has had its pasture manipulated)

    #. No annual pasture after two spraytopped pastures. Two spraytops will only be done prior to cropping. Tactical
       changes into more pasture can still be done after one spraytopped pasture.

    #. No resowing of pastures if the previous landuse is pasture and the following landuse is a crop.

    #. No non-spraytopped pasture between spraytopped pasture and crop. Spraytopping is in preperation for cropping so not going to stick a non-spraytop pasture inbetween.

    #. No spraytopped pasture between a crop and a non-spraytopped pasture. Wouldn't spraytop at the start of pasture phase (spraytopping may happen later in the rotation to tidy up pasture though)

    #. No resowing between spraytoping. No point spraytopping pasture then resowing it and spraying it out again.

    #. Perennials can only be in a continuous rotation (perennials are usually situated on soil that doesnt suit anything else)

    #. Only a single pasture variety in a rotation phase.

    #. Only canola after pasture (best use of soil conditions)

    #. No dry seeding after non-spraytopped pasture unless RR canola because of excessive weed burden.

    #. No saleable crop after fodder (due to excessive weed burden).



"""
import numpy as np
import pandas as pd
import itertools
from openpyxl import load_workbook

#AFO modules
from lib.AfoLogic import Functions as fun
from lib.AfoLogic import StructuralInputs as sinp
from lib.AfoLogic import PropertyInputs as pinp

# from . import Functions as fun
# from . import StructuralInputs as sinp
# from . import PropertyInputs as pinp

'''
Trevethan version 1:
Trevethan have cont pasture (which is added at the bottom) and and resown spray topped vetch pasture.
In version 1 I am just saying the the resown vetch pasture has the same productivity as the cont annual pasture. 
The vetch pasture is spraytopped so it reduces chem costs and it also reduces fert requirement in the following crop. 


Note a2 (pasture with no cost) doesnt provide anything and it doesnt need to because it is only selected until the latest brk.
'''
##if you want to use a customised list of rotations this can be set to false - populate the array further down the module.
customised_rotations = False
def f_rot_gen(user_crop_rot=False): #by default it runs the full rotation list
    print('Running rotation generator', end=' ')
    if not user_crop_rot:
        yr0 = np.array(['b', 'w',  'l', 'z','r','of'#,'lf', 'od', 'o','f', 'h'- not included in v1 to speed calibration process
                       , 'bd','wd','rd','zd'
                       , 'a'
                       ,'s'])#, 'm'
                        # , 'u', 'ur'
                        # , 'x', 'xr'
                        # , 'j', 't', 'jr', 'tr'])
        yr1 = np.array([
               'B','W', 'N', 'L1', 'OF'#'F',,'O1', 'LF'
               , 'A1'
               , 'S'])
               # , 'M'
                # , 'U'
                # , 'X'
                # , 'T', 'J'])
        yr2 = np.array(['B','O','W', 'N', 'L'#, 'F'
               , 'A'])# we dont track A2 because A2 is only required if S is being differentiated in yr2
               # , 'S' #for EWW we are not representing that two spraytopped pastures is better than one.
               # , 'M'
                # , 'U'
                # , 'X'
                # , 'T', 'J'])
        yr3 = np.array(['B','O','W', 'N', 'L'#, 'F'
               , 'A'])
                # , 'U'
                # , 'T'])
        yr4 = np.array(['A','Y'])
                # , 'U'
                # , 'T'])
        yr5 = np.array(['A','Y'])
                # , 'U'
                # , 'T'])

        # arrays=[yr5,yr4,yr3,yr2,yr1,yr0]
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
            phases = phases[~(np.isin(phases[:,i], ['N'])&np.isin(phases[:,i+1], ['N','r','z','rd','zd']))]
            ###no cont pulse - need this rule and the one below
            phases = phases[~(np.isin(phases[:,i], ['L','L1','LF','F'])&np.isin(phases[:,i+1], ['L','L1','LF','F','l','lf','f']))]
            ###two years between pulses
            if i<np.size(phases,1)-2:
                phases = phases[~(np.isin(phases[:,i], ['L','L1','LF','F'])&np.isin(phases[:,i+1], ['L','L1','LF','F','l','lf','f']))]
                phases = phases[~(np.isin(phases[:,i], ['L','L1','LF','F'])&np.isin(phases[:,i+2], ['L','L1','LF','F','l','lf','f']))]
            ###no pulse after pasture
            phases = phases[~(np.isin(phases[:,i], ['AR', 'SR1','A1','A2','A','M','S','S1','U','X','T','J'])&np.isin(phases[:,i+1], ['L','L1','LF','F','l','lf','f']))]
            ###no pasture after pulse
            phases = phases[~(np.isin(phases[:,i], ['L','L1','LF','F','l','lf','f'])&np.isin(phases[:,i+1], ['a', 's','A1','A2','A','M','S','S1','U','X','T','J']))]
            ###no fodder after pulse
            phases = phases[~(np.isin(phases[:,i], ['L','L1','LF','F','l','lf','f'])&np.isin(phases[:,i+1], ['of', 'OF']))]
            ###only spraytopped pasture after manipulated
            phases = phases[~(np.isin(phases[:,i], ['M'])&np.isin(phases[:,i+1], ['AR', 'A1','A2','A', 'M','a','ar','m']))]
            ###not going to resown tedera after a tedera (in a cont rotation you resow every 10yrs but that is accounted for with 'tc')
            phases = phases[~(np.isin(phases[:,i], ['T','J'])&np.isin(phases[:,i+1], ['tr','jr']))]
            ###not going to resow lucerne after a lucerne (in a cont rotation you resow every 5yrs but that is accounted for with 'uc' & 'xc')
            phases = phases[~(np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['xr','ur']))]
            ###only crop after two spraytopped pastures
            if i<np.size(phases,1)-2:
                phases = phases[~(np.isin(phases[:,i], ['S','S1','SR1'])&np.isin(phases[:,i+1], ['S','S1','SR1'])&np.isin(phases[:,i+2], ['AR','A1','A2','A','M','S','S1','SR1','ar','a','m','s','sr']))]
            ###No resowing of pastures if the previous landuse is pasture and the following landuse is a crop
            if i<np.size(phases,1)-2:
                phases = phases[~(np.isin(phases[:,i], ['AR','A1','A2','A','M','S','S1','SR1'])&np.isin(phases[:,i+1], ['AR','SR1'])&np.isin(phases[:,i+2], ['B','O','O1','W','N','L','L1','LF','F','OF','b','h','o','of','w','f','l','lf','z','r','bd','wd','rd','zd']))]
            ###No non spraytopped pasture between spraytopped pasture and crop
            if i<np.size(phases,1)-2:
                phases = phases[~(np.isin(phases[:,i], ['S','S1','SR1'])&np.isin(phases[:,i+1], ['AR','A1','A2','A'])&np.isin(phases[:,i+2], ['B','O','O1','W','N','L','L1','LF','F','OF','b','h','o','of','w','f','l','lf','z','r','bd','wd','rd','zd']))]
            ###No spraytopped pasture between a crop and a non spraytopped pasture
            if i<np.size(phases,1)-2:
                phases = phases[~(np.isin(phases[:,i], ['Y','B','O','O1','W','N','L','L1','LF','F','OF'])&np.isin(phases[:,i+1], ['S','S1','SR1'])&np.isin(phases[:,i+2], ['AR','A1','A2','A','M','ar','a','m']))]
            # ###No resowing between spraytoping
            # phases = phases[~(np.isin(phases[:,i], ['S','S1','SR1'])&np.isin(phases[:,i+1], ['SR1','sr']))]
            # ###only canola after pasture
            # phases = phases[~(np.isin(phases[:,i], ['AR','SR1','A1','A2','A','M','S','S1','U','X','T','J'])&np.isin(phases[:,i+1], ['B','O','O1','W', 'L','L1','LF', 'F', 'OF', 'b', 'h', 'o', 'of', 'w', 'f', 'l','lf', 'bd','wd']))]
            ###no dry seeding after non spraytopped pasture unless RR canola
            phases = phases[~(np.isin(phases[:,i], ['A1','A2','A','AR','M','U','X','T','J'])&np.isin(phases[:,i+1], ['bd','od','wd','zd']))]
            # ###no saleable crop after strategic fodder
            # phases = phases[~(np.isin(phases[:,i], ['OF'])&np.isin(phases[:,i+1], ['b', 'h', 'o', 'w', 'f', 'l','lf', 'z','r','bd','wd','rd','zd']))]
            ###can't have 1yr of perennial unless it is the earliest yr in the history
            if i == 0:
                pass #first yr of rotation can be a perennial because a  perennial could have been before it
            else:
                try: #used for conditions that are concerned with more than two yrs
                    phases = phases[~(np.isin(phases[:,i], ['T','J'])&~(np.isin(phases[:,i-1], ['T','J', 'Y']) + np.isin(phases[:,i+1], ['T','J','t','j'])))]
                except IndexError: pass
                ###can't have 1yr of perennial
                try: #used for conditions that are concerned with more than two yrs
                    phases = phases[~(np.isin(phases[:,i], ['U','X'])&~(np.isin(phases[:,i-1], ['U','X', 'Y']) + np.isin(phases[:,i+1], ['U','X','u','x'])))]
                except IndexError: pass

        ##lucerne and tedera resowing
        phases = phases[~(~np.isin(phases[:,np.size(phases,1)-2], ['U','X'])&np.isin(phases[:,np.size(phases,1)-1], ['U','X','u','x']))] #lucerne after a non lucern must be resown
        phases = phases[~(~np.isin(phases[:,np.size(phases,1)-2], ['T','J'])&np.isin(phases[:,np.size(phases,1)-1], ['T','J','t','j']))] #Tedera after a non tedera must be resown

        ##annual resowing
        resow_cols = np.size(phases,1)-resow_a #the number of cols where resowing can occur ie in yr0 and 1
        for i in range(resow_cols):
            i+=1
            ###if continuous annual it must be spraytopped every 5yrs
            a_index = np.all(np.isin(phases[:,np.size(phases,1)-i-resow_a:np.size(phases,1)-i], ['AR','A1','A2','A','M']), axis=1)&np.isin(phases[:,np.size(phases,1)-i], ['AR','A1','A2','A','M','ar','a','m'])
            phases = phases[~a_index]


            #there is no ar. Resowing is now controlled in the sim inputs.
            # ###if there are no annuals in the history then an annual in yr0 or yr1 must be resown
            # a_index2 = np.all(~np.isin(phases[:,np.size(phases,1)-i-resow_a:np.size(phases,1)-i], ['AR','SR1','A1','A2','A','M','S','S1']), axis=1)&np.isin(phases[:,np.size(phases,1)-i], ['a', 's','m','A1','A2','A','M','S','S1'])
            # phases = phases[~a_index2]

        cont_annual = np.array([['S', 'A', 'A', 'A1', 'a']
                               , ['A', 'S', 'A', 'A1', 'a']
                               , ['A', 'A', 'S', 'A1', 'a']
                               ])
        phases = np.concatenate((phases, cont_annual))

        ##X can't be in the same rotation as U, T, J and A
        a_xutj = np.any(np.isin(phases[:,:], ['AR', 'SR1','ar','a','A1','A2','A','m','M','s','sr','S','S1']), axis=1)&np.any(np.isin(phases[:,:], ['X','x','xr','U','u','ur','T','t','tr','J','j','jr']), axis=1)
        x_utj = np.any(np.isin(phases[:,:], ['X','x','xr']), axis=1)&np.any(np.isin(phases[:,:], ['U','u','ur','T','t','tr','J','j','jr']), axis=1)
        u_tj = np.any(np.isin(phases[:,:], ['U','u','ur']), axis=1)&np.any(np.isin(phases[:,:], ['T','t','tr','J','j','jr']), axis=1)
        t_j = np.any(np.isin(phases[:,:], ['T','t','tr']), axis=1)&np.any(np.isin(phases[:,:], ['J','j','jr']), axis=1)
        phases = phases[~(a_xutj + x_utj + u_tj + t_j)]


        ######################################################
        #delete cont rotations that require special handling#
        ######################################################
        '''
        remove cont rotations before generalisation
        -remove rotations that can provide themselves
        -when generalising don't generalise yr1 
            - this will stop Y X X X3 x4 providing itself ie if you generalised yr1 to X then this rotation would provide itself'
        '''
        ##check if every phase in a rotation is either lucerne or Y
        xindex=np.all(np.isin(phases[:,:], ['X','x','U','u']), axis=1)
        phases = phases[~xindex]
        ##check if every phase in a rotation is either T or Y
        tindex=np.all(np.isin(phases[:,:], ['T','J','t','j']), axis=1)
        phases = phases[~tindex]

        ########################
        #continuous phase & pnc#
        ########################
        sp=np.array(['SP','SP','SP','SP','sp'])
        phases = np.concatenate((phases, [sp])) #square brackets required because otherwise it thinks that the cont rotations are just 1D

        pnc=np.array([['G','G','G','Ag1','a2']
                     ,['G','G','G','Sg1','a2']
                     ,['G', 'G', 'G', 'U', 'a2']  # can only have multiple Lucerne
                     ,['G','G','Ag2','N','a2']
                     ,['G','G','Ag2','E','a2']
                     ,['G','G','Ag2','OF','a2']
                     ,['G','G','Ag2','LF','a2']
                     ,['G','G','C1','N','a2']
                     ,['G','G','C1','E','a2']
                     ,['G','G','C1','P','a2'] #Note APa doesn't exist therefore only necessary option is CPa
                     ,['G','G','C1','OF','a2']
                     ,['G','G','C1','LF','a2']
                     ])
        phases = np.concatenate((phases, pnc))



        # todo add back in when required.
        # tc=np.array(['tc','tc','tc','tc','tc'])
        # phases = np.concatenate((phases, [tc])) #square brackets required because otherwise it thinks that the cont rotations are just 1D
        #
        # jc=np.array(['jc','jc','jc','jc','jc'])
        # phases = np.concatenate((phases, [jc])) #square brackets required because otherwise it thinks that the cont rotations are just 1D
        #
        # uc=np.array(['uc','uc','uc','uc','uc'])
        # phases = np.concatenate((phases, [uc])) #square brackets required because otherwise it thinks that the cont rotations are just 1D
        #
        # xc=np.array(['xc','xc','xc','xc','xc'])
        # phases = np.concatenate((phases, [xc])) #square brackets required because otherwise it thinks that the cont rotations are just 1D

    ######################
    #simplified rotations#
    ######################
    '''
    There are two options for simplified rotations:
    1. enter them manually below (this can be good for testing purposes)
    2. enter them in property.xl in the crop sheet (this is the more common option)
    
    '''

    ##option 1: enter the rotaions you want represented in the model
    ###This bit of code can be used to extract custom rotations only.
    ###You can alter the user_rot array to include to rotations you want to include (all other rotations will be excluded)
    ###The sequence of the rotation is irrelevant ie b b w w is the same as w b b w.
    ds_user_rot_init = np.array([['ar', 'a', 'w', 'w', 'r', 'b']
                                 ,['r', 'w', 'b', 'r', 'w', 'b']])

    if customised_rotations:
        ##roll the rotation to make all the phases required for a given rotation
        ds_user_rot=ds_user_rot_init
        for offset in range(1,np.size(ds_user_rot_init,axis=1)):
            ds_user_rot = np.concatenate((ds_user_rot, np.roll(ds_user_rot_init, offset, axis=1)),axis=0)
        ##the next code simplifies the full list of phases to only include the necessary ones to represent the user rotations.
        ix_bool=np.zeros(len(phases))
        ###loop through all rotation phases then loop through all of the user rotations.
        ###this checks if the rotation phases are a superset of any of the user rotations
        for s_rot_phase, ix_phase in zip(phases,range(len(phases))):
            for s_user_rot in ds_user_rot:
                req=1
                for i in range(np.size(s_user_rot)):
                    req*=sinp.landuse[s_rot_phase[i]].issuperset({s_user_rot[i]})
                ix_bool[ix_phase]=max(req,ix_bool[ix_phase])
        phases=phases[ix_bool>0]


    ##option 2: if you want to represent the rotations from property.xlsx
    if user_crop_rot:
        phases = pinp.crop['fixed_rotphases'].reset_index().values.astype('str')
        ###initilise these so that rot generator can be checked (want to make sure each rotation provides a history other than pnc)
        pnc=np.array([['G','G','G','Ag1','a2']
                     ,['G','G','G','Sg1','a2']
                     ,['G', 'G', 'G', 'U', 'a2']  # can only have multiple Lucerne
                     ,['G','G','Ag2','N','a2']
                     ,['G','G','Ag2','E','a2']
                     ,['G','G','Ag2','OF','a2']
                     ,['G','G','Ag2','LF','a2']
                     ,['G','G','C1','N','a2']
                     ,['G','G','C1','E','a2']
                     ,['G','G','C1','P','a2'] #Note APa doesn't exist therefore only necessary option is CPa
                     ,['G','G','C1','OF','a2']
                     ,['G','G','C1','LF','a2']
                     ])


    ############################################################################################################################################################################################
    ############################################################################################################################################################################################
    #Generate the paramater for rotation provide and require
    ############################################################################################################################################################################################
    ############################################################################################################################################################################################


    ##history require
    rot_hist = phases[:,0:np.size(phases,1)-1]
    rot_hist = np.unique(rot_hist, axis=0)
    pnc_hist = pnc[:,0:np.size(phases,1)-1]

    ##generate a list of the phases and histories (agretated version)
    l_phases = [''.join(x) for x in phases.astype(str)]
    l_rot_hist = [''.join(x) for x in rot_hist.astype(str)]


    ##################
    #con 1 param     #
    ##################
    '''determines what history1 each rotation requires and provides'''
    mps_bool_req=[]
    mps_bool_prov=[]
    for rot_phase in phases:
        ##test variable just used to help check all rotations provide and require a history - only for user information.
        ##every rotation should provide and require a history.
        test=0
        test2=0
        for hist in rot_hist:
            l_hist=[]
            for i in range(len(hist)):
                l_hist.append(sinp.landuse[hist[i]]) #deterimines the sets in each constraint
            req=1
            prov=-1
            for i in range(len(hist)):
                req*=np.isin(rot_phase[i], list(l_hist[i]))  #checks each set in a given rotation for the req part of the equation
                prov*=np.isin(rot_phase[i+1],list(l_hist[i])) #checks each set in a given rotation for the prov part of the equation
            if not any(all(hist == h) for h in pnc_hist) or rot_phase[-1]=='a2': #For the error check, ignore pnc histories unless current landuse is a2 - otherwise rotations just provide pnc and we dont catch errors
                test+=prov
                test2+=req
            mps_bool_req.append(req)
            mps_bool_prov.append(prov)
        if test==0 and rot_phase[-1]!='a2': #doesn't provide a history. Note a2 (pasture with no cost) doesnt provide anything and it doesnt need to because it is only selected until the latest brk.
            print('rot does not provide a history: ',rot_phase)
        if test2==0: #doesn't require a history
            print('rot does not req a history: ',rot_phase)

    rot_phase_by_constrain = pd.DataFrame(list(itertools.product(l_phases,l_rot_hist))) #had to use this cartesian method as i couldn't get the fast function to work
    mps_bool_req = pd.Series(mps_bool_req) #convert to series because easier to manipulate
    mps_bool_prov = pd.Series(mps_bool_prov) #convert to series because easier to manipulate
    mps_bool_req = pd.concat([rot_phase_by_constrain, mps_bool_req], axis=1) #add two dfs together
    mps_bool_req.columns = [0,1,2]
    mps_bool_prov = pd.concat([rot_phase_by_constrain, mps_bool_prov], axis=1) #add two dfs together
    mps_bool_prov.columns = [0,1,2]
    mps_bool_req = mps_bool_req[(mps_bool_req.iloc[:,2] != 0)]
    mps_bool_prov = mps_bool_prov[(mps_bool_prov.iloc[:,2] != 0)]

    ##test to make sure all histories are required (this may not happen if the set definitions are wrong)
    if len(np.unique(mps_bool_req.iloc[:,1].values)) != len(l_rot_hist):
        hist_req = np.unique(mps_bool_req.iloc[:, 1].values) #all the histories that are required.
        hist_errors = np.array(l_rot_hist)[np.all(np.array(l_rot_hist)!=hist_req[:,None],axis=0)]
        raise ValueError(f'Histories are not all required. This indicates error with set definitions. Histories not required: {hist_errors}' )

    ##remove rotations that require a history that is not provided by any rotation.
    ## This is requireed for phases like YNOS1z that cant be provided due to needing an annual in yr5 but only
    ## canola can follow annual but yr3 is cannola so yr4 cant be canola
    ###if it is user rot then leave in all rotations even if they are not provided because otherwise inputs will not line up
    if user_crop_rot:
        rot_phases =  pd.DataFrame(phases, index=l_phases)
        rot_hist = pd.DataFrame(rot_hist, index=l_rot_hist)
    else:
        ###get a list of the histories that are provided by the rotations
        hist_prov = np.unique(mps_bool_prov.iloc[:,1].values)
        ###get a list of any of the rotations that require a not provided history - need to do it this way so that rotation is fully removed even if it requires two histories
        phases_to_remove = np.unique(mps_bool_req[np.all(mps_bool_req.iloc[:, 1].values != hist_prov[:,None], axis=0)].iloc[:, 0].values)
        ###remove the rotations from the require param, provide param and the phase list
        mps_bool_req = mps_bool_req[np.all(mps_bool_req.iloc[:, 0].values!=phases_to_remove[:,None], axis=0)]
        mps_bool_prov = mps_bool_prov[np.all(mps_bool_prov.iloc[:, 0].values!=phases_to_remove[:,None], axis=0)]
        rot_phases =  pd.DataFrame(phases, index=l_phases)
        rot_phases = rot_phases[np.all(rot_phases.index.values!=phases_to_remove[:,None], axis=0)]
        rot_hist = pd.DataFrame(rot_hist, index=l_rot_hist)
        rot_hist = rot_hist[np.any(rot_hist.index.values==hist_prov[:,None], axis=0)]

    print('- finished')
    return {"phases_r": rot_phases, "rot_req": mps_bool_req, "rot_prov": mps_bool_prov, "s_rotcon1": rot_hist}



if __name__ == '__main__': #use this so that sphinx doesn't run all the code when generating the docs
    import warnings
    from lib.RawVersion import LoadExcelInputs as dxl
    from lib.RawVersion import LoadExp as exp
    from lib.AfoLogic import relativeFile

    exp_data, exp_group_bool, trial_pinp = exp.f_read_exp()
    sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)
    sinp.f_select_n_reset_sinp(sinp_defaults)
    sinp.f_landuse_sets()
    d_rot_info = f_rot_gen()
    rot_phases = d_rot_info["phases_r"]
    mps_bool_req = d_rot_info["rot_req"]
    mps_bool_prov = d_rot_info["rot_prov"]
    rot_hist = d_rot_info["s_rotcon1"]

    ##start writing
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
























