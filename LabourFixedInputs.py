# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:42:51 2019

module: labour fixe input module - contains all input data likely to vary for different regions or farms

extra info: - this could eventually interact with a user interface
            - interacts with kv's

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
     
@author: young
"""
import datetime
#######################################
#define dict for labour fixed  inputs #
#######################################

labour_fixed_input_data = dict()

'''
labour fixed
'''

###########
#learn    #
###########

#difference to old midas - i am now allowing the model to choose the labour periods for learning, to represent that labourers would study in less busy periods
#                        - there is just one input number to represent labour learn, at the time of writing this number represents 6hrs for each crop and pasture and grazing method
#                        - the old midas used to include and not inclued learning abour phased ie lucerne depending if it was included as an option but i think farmer are still learning about things even if not currently doing it. so thats why this is now just one value 
#                        - although maybe it could be broken up a little to make it easier to calibrate 
#                        - i think it should also include time the workers spend consulting with farm advisors etc 
labour_fixed_input_data['labour_learn'] = 66 


######
#tax #
######

#when inputting check dates line up correctly with labour periods ie you may not want any fixed labour during seeding and harvest
                                                #date    #number of hours
labour_fixed_input_data['labour_tax'] = {datetime.datetime(2019,3,1)	: 6,   \
                            datetime.datetime(2019,4,1)  : 10,  \
                            datetime.datetime(2019,6,15)  : 5}

###########
#planning #
###########

#when inputting check dates line up correctly with labour periods ie you may not want any fixed labour during seeding and harvest
                                                #date    #number of hours
labour_fixed_input_data['labour_planning'] = {datetime.datetime(2019,1,1)	: 4,   \
                                datetime.datetime(2019,2,1)	: 4,   \
                                datetime.datetime(2019,3,1)	: 4,   \
                                datetime.datetime(2019,4,1)	: 4,   \
                                datetime.datetime(2019,5,1)	: 4,   \
                                datetime.datetime(2019,6,15)	: 4,   \
                                datetime.datetime(2019,7,1)	: 4,   \
                                datetime.datetime(2019,8,1)	: 4,   \
                                datetime.datetime(2019,9,1)	: 4,   \
                                datetime.datetime(2019,10,1)  : 4,   \
                                datetime.datetime(2019,11,1)  : 4}

######
#bas #
######

#when inputting check dates line up correctly with labour periods ie you may not want any fixed labour during seeding and harvest
#current dates are inaccordance with bas due date
                                                #date    #number of hours
labour_fixed_input_data['labour_bas'] = {datetime.datetime(2019,2,1)	: 20,   \
                            datetime.datetime(2019,4,1)  : 20,  \
                            datetime.datetime(2019,7,1)	: 20,   \
                            datetime.datetime(2019,10,1) : 20}

#############
#super & WC #
#############

#when inputting check dates line up correctly with labour periods ie you may not want any fixed labour during seeding and harvest
                                                #date    #number of hours
labour_fixed_input_data['labour_super'] = {datetime.datetime(2019,3,1)	: 10,   \
                            datetime.datetime(2019,6,15)  : 10,  \
                            datetime.datetime(2019,9,1)	: 10,   \
                            datetime.datetime(2019,12,15)  : 10}
