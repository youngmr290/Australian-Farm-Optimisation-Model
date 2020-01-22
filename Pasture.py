# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:15:40 2019

Version Control:
Version     Date        Person  Change
   1.1      30Dec19     JMY     Added timing to check elapased time for each function in pasture functions

Known problems:
Fixed   Date    ID by   Problem
        

@author: John
"""

import PastureFunctions as pf

from timeit import default_timer as timer
time_list = []
time_was = []
time_list.append(timer()) ; time_was.append("start")


annual = pf.PastDetailed('annual', {'a', 'ar', 'a3', 'a4', 'a5', 's', 'sr', 's3', 's4', 's5', 'm', 'm3', 'm4'},'Property.xlsx')        # create an instance of the Pasture class and pass the landuse name and the filename for the Excel file that stores the data
time_list.append(timer()) ; time_was.append("define annual")

annual.read_inputs_from_excel()                         # read inputs from Excel file and map to the python variables
time_list.append(timer()) ; time_was.append("read inputs from Excel")

annual.calculate_germination()                          # calculate the germination for each rotation phase
time_list.append(timer()) ; time_was.append("germination")

annual.calculate_reseeding()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("reseeding")

annual.dry_feed()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("dry feed")

annual.green_feed()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("green feed")

annual.poc_con()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("poc con")

annual.poc_md()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("poc_md")

annual.poc_vol()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("poc_vol")




#report the timer results
time_prev=time_list[0]
for time_step, time in enumerate(time_list):
    time_elapsed = time-time_prev
    if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
    time_prev=time
print("elapsed total time for pasture module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in secondsfirst


#test times
#def test1():
#    annual.germ_phase_data.columns.values[range(phase_len)] = [*range(phase_len)]
#def test2():
#    annual.germ_phase_data.columns.values[0:phase_len] = [*range(phase_len)]
#    
#print(timeit.repeat(test1,number=5,repeat=10))
#print(timeit.repeat(test2,number=5,repeat=10))    
