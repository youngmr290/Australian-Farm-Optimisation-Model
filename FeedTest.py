# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

A module to create a table of the NV for increments of the feed supply and the dates of the FVPs
The output is used in the [Feed supply calculator.xlsx] to generate the optimum profiles in an iterative process.

@author: John
"""
import pandas as pd
import numpy as np

import FeedSupplyGenerator as fgen
#import Functions as fun

print("")
print("It is good to run FeedTest with a temporary Structural.xls with maximum age set to 9.5year & 5 FVPs")
print("This generates the dams & offspring periods for the full width of the FeedSupply spreadsheet")
print("")

na = np.newaxis

## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('Output/r_NV.xlsx',engine='xlsxwriter', datetime_format="dd-mmm-yy")
### Set some standard values for the workbook
workbook = writer.book
format_0hidden2dp = workbook.add_format({'num_format': '# ##0.00;-0.00;;@'})
format_header = workbook.add_format({
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'border': 2})


# the final result has the following axes
## p6 - axis0: the feedsupply period
## f - axis1: the levels of the feed supply. There are 30 slices being the values from 0 to 3 in 0.1 increments
## j - axis1 alternative: the levels of the feed supply. The slices are feedsupply level 0: between 0 and 1, 1: between 1 and 2, 2: between 2 and 3
## i - axis2: the index or power of the polynomial

# call the feedsupply generator that returns the NV for each feedsupply between 0 and 3
r_nv_p6f, feedsupply_f = fgen.feed_generator()
# reset last feedsupply to 3 for fitting the polynomial
feedsupply_f[-1] = 3
# concatenate the feedsupply & the NV into a single array and then convert to a dataframe for saving to Excel
temp = np.concatenate((feedsupply_f[na,:],r_nv_p6f), axis=0)
data_df1 = pd.DataFrame(temp)

## call the period generator that returns the NV for each feedsupply between 0 and 3
date_start_p, fvp_fdams, fvp_foffs, a_p6_pz = fgen.period_generator()
## reduce dimension of the fvp arrays to 2 dimensions
# todo this needs to be altered if shape of the inputs is altered
### common for dams & offs
z_slc = 0
i_slc = slice(None)
### dams
###Select e axis for dams
e1_dams_slc = 0
###offspring
###Select d & x slice for offspring
d_offs_slc = 2
x_offs_slc = 0

## convert each to a dataframe for saving to Excel
data_df2 = pd.DataFrame(date_start_p)
data_df3 = pd.DataFrame(a_p6_pz)

### alter the axes that are added to the dataframe based axes that will vary. Also change the columns in the writer below
fvp_fdams = fvp_fdams[:, :, e1_dams_slc, :, :, :, z_slc, i_slc, :, :, :, :, :, :, :]
fvp_foffs = fvp_foffs[:, :, :, :, :, :, z_slc, i_slc, d_offs_slc, :, :, :, x_offs_slc, :, :]

### Use squeeze to highlight (with an error) that the number of active axes has changed.
data_df4 = pd.DataFrame(np.squeeze(fvp_fdams).astype('datetime64[ns]'))  # conversion to dataframe only works with this datatype
data_df5 = pd.DataFrame(np.squeeze(fvp_foffs).astype('datetime64[ns]'))




## write the data and the polynomials to Excel (overwriting file r_nv.xlsx)
### aim to keep a blank column between the output (to highlight when the data has over run)
data_df1.to_excel(writer, 'NV', index=False, header=False, startrow=0, startcol=0)

##The period date dataframe
first_col_df2 = 0
data_df2.to_excel(writer, 'Periods', index=True, header=False, startrow=1, startcol=first_col_df2)

##The feed periods dataframe
first_col_df3 = 3
data_df3.to_excel(writer, 'Periods', index=False, header=False, startrow=1, startcol=first_col_df3)

##The FVP dates for the dams
first_col_df4 = 5
data_df4.to_excel(writer, 'Periods', index=False, header=False, startrow=1, startcol=first_col_df4)

##The FVP dates for the offs
first_col_df5 = 8
data_df5.to_excel(writer, 'Periods', index=False, header=False, startrow=1, startcol=first_col_df5)

##Write column headers with format   todo Work out how to add a column heading to each df
worksheet = writer.sheets['Periods']
###df2   +1 on col_num because index=True
for col_num, value in enumerate(data_df2.columns.values):
    worksheet.write(0, first_col_df2 + col_num + 1, value, format_header)
###df3
for col_num, value in enumerate(data_df3.columns.values):
    worksheet.write(0, first_col_df3 + col_num, value, format_header)
###df4
for col_num, value in enumerate(data_df4.columns.values):
    worksheet.write(0, first_col_df4 + col_num, value, format_header)
###df5
for col_num, value in enumerate(data_df5.columns.values):
    worksheet.write(0, first_col_df5 + col_num, value, format_header)


# # set some values required for the polynomials
# len_p6 = r_nv_p6f.shape[0]
# len_j = 3
# max_order = 5
# r2_target = 0.9995
#
# # loop through decreasing levels of polynomials (to allow selection of the minimum order that adequately fits the data for each period (p6))
# ## need a different polynomial for the feedsupply ranges 0 to 1, 1 to 2 and 2 to 3. With a different slice range for the f axis
# poly_coeff = np.zeros((len_p6, len_j, max_order+1))
# for j in range(len_j):
#     ### set the slice of the feed supply for this j
#     start = j * 10
#     stop = (j+1) * 10 +1
#     slc = slice(start, stop)
#
#     ### calculate the total sum of squares for the slice range for each p6
#     #### calculate y bar for each p6
#     ybar = np.mean(r_nv_p6f[:,slc].T, axis = 0)
#     #### calculate the total sum of squares
#     tss = np.sum(np.square(r_nv_p6f[:,slc].T - ybar), axis = 0)
#
#     ## loop down through the order of the polynomial
#     for order in range(max_order,0,-1):
#         ### fit the polynomial and return the full results so that r2 can be calculated
#         polyresults = np.polyfit(feedsupply_f[slc], r_nv_p6f[:,slc].T, order, full=True)
#         rsquared_p6 = 1 - (polyresults[1] / tss)
#         ###store the polynomial coefficients if r2 is greater than the target
#         ####access the coefficients
#         shortlist = polyresults[0].T
#         #### store the results if r2 is above the target
#         #### align the coefficients for this order of the polynomial with the maximum
#         ### set the slice of the coefficients for this order. The slices are x**4, x**4, x**3, x**2, x, intercept
#         temporary = np.zeros((len_p6, max_order+1))
#         start = max_order - order
#         stop = max_order + 1
#         slc = slice(start, stop)
#         temporary[:,slc] = shortlist
#
#         poly_coeff[:,j,:] = fun.f_update(poly_coeff[:,j,:], temporary, rsquared_p6[:,na] >= r2_target)
#
#     poly_coeff_df = pd.DataFrame(poly_coeff[:,j,:])
#     ## write the data and the polynomials to Excel (overwriting file r_nv.xlsx)
#     sheetname = "Coeff" + str(j)
#     poly_coeff_df.to_excel(writer, sheetname, index=False, header=False)

writer.save()

print("Feed Generator complete, r_NV.xlsx has been created")
