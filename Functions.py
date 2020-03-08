# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:29:17 2019

module: functions module - contains all the core functions that we have made

Version Control:
Version     Date        Person  Change
1.1         10Dec19     John    xl_all_named_ranges: Commented the updates made
                                                     removed cells temporary variable and parameters dict is updated directly from the cells read
1.2         11Dec19     John    xl_all_named_ranges: Added handling of rangename errors. Pass over index errors that are associated with 'bad' range names
                                                     IndexError handles when a sheet with names has been deleted (ie a sheet_name error)
                                                     TypeError handles a name in the target sheet is #REF (ie a cell_range error)
1.3         12Dec19     MRY     phases: Altered the phase filter to compare the landuse in the following pairs (0,1),(1,2)...(len_phase-2,len_phase-1) so the first and last year are not compared
1.4         13Dec19     MRY     added cartesian_product_simple_transpose - a fast func for making every possibility of multiple lists
1.5         22Dec19     John    period_allocation: simplify the function so that it doesn't redefine variable from the parameters passed
                                range_allocation: added this function fashioned from period_allocation
1.6         26Dec19     JMY     xl_all_named_ranges: altered 2 comments that were in the wrong position
1.7         19Jan20     MRY     alted cost period function to handle df with undefined title - because now inputs are read in from excel the column name can vary, which it couldn't before because the df was built from dict hence colum name was always 0


Known problems:
Fixed   Date    ID by   Problem


@author: young
"""
import pandas as pd
import timeit
import numpy as np
from dateutil.parser import parse
import itertools

#this module shouldn't import other midas modules

################################################
#function to read in excel named ranges to a df#
################################################
#requires being passed the filename and sheetname for the workbook that will be accessed
#returns a dict with the key being the excel rangename
#the dict includes: numbers (where the rangename is a single cell), lists (where the rangename is one dimensional) and dataframes (where the range is 2 dimensional)
#If the range is 2D the function converts the first row to the dataframe column names and the first col to index names
#if you dont want this you can reset index using index.reset or something and probs the similar for cols
#Testing shpwed readonly = False was quicker than true. But still not as fast as pandas
# (may not exist anymore) now it causes problems somoetimes locking you out of excel because it is readonly - closing doesn't fix issue (wb._archive.close())

def xl_all_named_ranges(filename, targetsheets, rangename=None):     # read all range names defined in the list targetsheets and return a dictionary of lists or dataframes
    ''' Read data from named ranges in an Excel workbook.

    Parameters:
    filename is an Excel worbook name (including the extension).
    targetsheets is a list of (or a single) worksheet names from which to read the range names.
    rangename is an optional argument. If not included then all rangenames are read.
    If included only that name is read in.

    Returns:
    A dictionary that includes key that correspond to the rangenames
    '''
    from openpyxl import load_workbook
    from openpyxl.worksheet.cell_range import CellRange

    wb = load_workbook(filename, data_only=True, read_only=False)
    # t_wb = wb
    parameters = {}
    ## convert targetsheets to lowercase and handle both an individual name and a list
    try:
        targetsheets = targetsheets.lower()
    except:   #targetsheets is a list
        targetsheets = [name.lower() for name in targetsheets]

    for dn in wb.defined_names.definedName[:]:
        if rangename is None or dn.name == rangename:
            try:
                sheet_name, cell_range = list(dn.destinations)[0]        # if it is a non-contiguous range dn.destinations would need to be looped through
                #print (dn.name, cell_range)
                if sheet_name.lower() in targetsheets:     # in to check list of sheet names
                    try:
                        cr = CellRange(cell_range)
                        width = cr.max_col - cr.min_col
                        length = cr.max_row - cr.min_row
                        ws = wb[sheet_name]
                        #print (dn.name, sheet_name, cell_range, length, width)
                        if not width and not length:            # the range is a single cell & is not iterable
                            parameters[dn.name] = ws[cell_range].value
                        elif not width:                         # the range is only 1 column & is not iterable across the row
                            parameters[dn.name] = np.asarray([cell.value for cell in [row[0] for row in ws[cell_range]]])
                        elif not length:                        # the range is 1 row & is iterable across columns
                            for row in ws[cell_range]:
                                parameters[dn.name] = np.asarray([cell.value for cell in row])
                        else:                                   # the range is a region & is iterable across rows and columns
                            df = pd.DataFrame([cell.value for cell in row] for row in ws[cell_range])
                            #df = pd.DataFrame(cells)
                            #print(df)
                            df.rename(columns=df.iloc[0],inplace=True)
                            ## drop row that had header names (renaming is more like a copy than a cut)
                            df.drop(df.index[0],inplace=True)
                            df = df.set_index(df.iloc[:,0]) #could use rename ie df.rename(index=df.iloc[:,0],inplace=True)
                            ## now have to drop the first col because renaming/set_index is more like copy than cut hence it doenst make the index col one just rename index to match col one
                            df = df.drop(df.columns[[0]],axis=1) #for some reason this will chuck an error in the index values are int and there is nothing in the top left cell of the df...seems like a bug in python
                            ## manipulate data into cheapest format - results in mainly float32 (strings are still objects) - without this each value is treated as an object (objects use up much more memory) - this change reduced fert df from 150mbs to 20mbs
                            parameters[dn.name] = df.apply(pd.to_numeric, errors='ignore', downcast='float')
                    except TypeError:
                        pass
            except IndexError:
                pass
    wb.close
    return parameters #t_wb #

#def test():
#    sheettest = xl_all_named_ranges("GSMInputs.xlsx","Annual") #sheettest = xl_all_named_ranges("GSMInputs.xlsx","Annual", True)
#def test1():
#    sheettest = xl_all_named_ranges("GSMInputs.xlsx","Annual")  #sheettest = xl_all_named_ranges("GSMInputs.xlsx","Annual", False)
##exceldata = xl_all_named_ranges('GSMInputs.xlsx', 'Annual')
##print(exceldataddd)
#crop_input = dict()
#crop_input['excel_ranges'] = {'crop inputs.xlsx':(['yield'            #yeild t/ha will be converted to kg in prep calcs
#                                              ,'frost'              #frost by crop by lmu (% yeildc reduction)
#                                              ,'yield_by_lmu'      #yeild by soil table
#                                              ,'seeding_rate'       #seeding rate by crop by lmu
#                                              ,'fert'               #fert t/ha for each crop includes 1yr previous phase
#                                              ,'fert_by_lmu'
#                                              ,'arable'
#                                              ,'passes'])}    #fert soil factor
#def test2():
#    xl_named_range(crop_input['excel_ranges'])
#
#print('All ranges True', min(timeit.repeat(test,number=3,repeat=5))/3)
#print('All ranges False', min(timeit.repeat(test1,number=3,repeat=5))/3)

# ########################
# #phases                #
# ########################
# #makes a df of all possible rotation phases
# #use product function to do a Cartesian product.
# def phases(landuses,phase_number):
#     phases = [landuses]*phase_number
#     df = pd.DataFrame(list(itertools.product(*phases) ) ) # '*' is used to unpack lists into multiple args
#     #function to remove unrealistic phases using some rules
#     #not comparing beginning phase with end phase because that ramoves the possibility of longer rotations ie nwbnwbnwb
#     #you cant have nwbnnwbn because the rotation phase bnnw is cut out by section below. But nwbn is still relevant as shown above.
#     cols=list(df.columns)
#     for i in range(len(cols)-1):
#         df = df.loc[-((df[cols[i]].isin(['rcanola','tcanola']))&(df[cols[i+1]].isin(['rcanola','tcanola'])))] #no cont canola
#         df = df.loc[-((df[cols[i]].isin(['lupins','faba']))&(df[cols[i+1]].isin(['lupins','faba'])))] #no cont faba or lupins ie lup lup, faba faba or lup faba etc
#         df = df.loc[-((df[cols[i]].isin(['barley','oats','fodder','hay']))&(df[cols[i+1]].isin(['wheat'])))] #wheat would be the first cereal in a rotation
#     return df#.astype('category') - cant do this because then merge doesn't work #use category to reduce size:  https://www.dataquest.io/blog/pandas-big-data/    (i was hoping this would speed up my big df stacking but it didnt make a huge difference)

#this is the fastest function for building cartesian products. Doesn't make much diff for small ones but upto 50% faster for big ones
def cartesian_product_simple_transpose(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

#print(timeit.timeit(phases2,number=100)/100)
#
#
##########################
# period calculators     #
##########################

def period_allocation(period_dates,periods,start,length=None):
    '''
    Parameters
    ----------
    period_dates : List
        Dates of the periods you are matching within eg labour periods or cashflow periods
    periods : List
        Name of each period.
    start : Date
        Date of interest.
    length : Dt, optional
        Length of the period of interest. The default is ''.

    Returns
    -------
    Proportion of a given date range in each period:
    Either take a date and returns the period it is in
    or take a date and a length and return a dataframe with a proportion in each period

    '''
    #gets the dates
    # period_dates = p_dates   # don't need this step if the variables passed in are changed to period_dates from p_dates and periods from p_name
    #gets the period name
    # periods = p_name
    if length is not None:
    #start empty list to append to
        allocation_period = []
        end = start + length
        #check how much of the range falls into each cash period
        for i in range(len(periods)-1):
            ## ^might be simpler to do this with allocation_period.append  \
            ## # (min(per_end,end)-max(per_start,start)/(end-start)) clipped(0,1)
            ## # would also be quicker if the loop started with i = bisect(period_dates,start)-1
            ## # and finished when per_end > end
            ## # perhaps this is a while loop
            per_start= period_dates[i]
            per_end = period_dates[i + 1]
            #if the range lasts longer than one cashflow period then that cashflow period gets allocated a proportion
            if start <=  per_start and end >= per_end:
                 allocation_period.append((per_end - per_start) / (end - start))
            #start of the range is before period and the end of the range is after the start of the period but before the end
            elif start <=  per_start <= end and end <= per_end:
                allocation_period.append((end - per_start) / (end - start))
            #is the start of the range after the start of the period and before the finish of the period
            #and the end of the range is after the end of the period
            elif start >=  per_start and start <= per_end <= end :
                allocation_period.append((per_end - start) / (end - start))
            #if all of the range occurs within one the period
            elif start >=  per_start and end <= per_end:
                allocation_period.append(1)
            #if the range doesn't occur in a period.
            else:
                allocation_period.append(np.nan)
        return pd.DataFrame(list(zip(periods,allocation_period)), columns= ('period', 'allocation'))
    #returns the period name a given date falls into
    else:    #^ could use the python function allocation_p = bisect.bisect(period_dates,start)-1
        for date, period in zip(period_dates, periods):
            while date <= start:
                allocation_p = period
                break
        return allocation_p


#input one or more df that have a date as a key and a value that you want to add to a period
#must also input the period dates and names as a list (these get used in the func above which is called by this)
#function returns a new dict with the sum total of all the values from the input dicts in the corresponding period they were compared to
#example of this func is in the labourcrop module
def df_period_total(p_dates,p_name,*dfs):
    #create empty numpy array that i will add the labour time in each period to
    array = np.zeros(len(p_name))
    #have to loops to allow for passing in multiple dicts
    for d in dfs:
#        print(dic)
        for date in d.index:
            # date = parse(key, dayfirst = False) #parse - simple way to go from string to datetime
            labour_period = period_allocation(p_dates,p_name,date)
            array[labour_period] += d.loc[date,d.columns]
    return dict(enumerate(array))



######################
# functions below are used to manipulate the period allocation from the func above into neccessary format
#they depend on the input type ie dict of df, and the number of entries
######################

def period_allocation_reindex(df, p_dates, p_name, start, length):
    '''
    Parameters
    ----------
    df : Dataframe
        1d dataframe or series.
    p_dates : List
        Dates of the period you are matching ie cashflow or labour.
    p_name : List
        Names of the period you are matching ie cashflow or labour.
    start : Datetime
        Start date of the df activity ie start date of seeding.
    length : Datetime
        Length of the df activity ie length of seeding.

    Returns
    -------
    Dataframe 2D
        This function is used when multiple activities have the same period allocation.
        - this func basically just calls the main allocation func then reindexes the df so it applies to all activities.
        eg the cost of seeding for each lmu is incurred over the same period so we just want to return a df with a certain cost in it for each lmu and each cashflow period (the cost may differ but the allocation will be the same for each lmu, so the allocation func is called once then reindexed then multiplied by the differnt costs)
    '''
    allocation = period_allocation(p_dates, p_name,start,length)
    allocation = allocation.set_index('period')
    columns = pd.MultiIndex.from_product([allocation.columns, df.index])
    allocation = allocation.reindex(columns,axis=1,level=0) #add level so mul can happen
    allocation.columns = allocation.columns.droplevel(0) #drop added level
    # cost = df.rename(index={0:'allocation'}).stack()
    df = allocation.mul(df.iloc[:,0],axis=1)
    return df

#this func returns a df with index name =
#column names = keys from dict you entered start date with
#input two dicts one with the start date and one with the length of the activity
def period_allocation2(start_df, length_df, p_dates, p_name):
    '''
    Parameters
    ----------
    start_df : Datetime series
        Contains the activity start dates ie start date of fert spreading for each fert.
    length_df : Datetime series
        Length of the df activity ie length of fert spreading for each fert.
    p_dates : List
        Dates of the period you are matching ie cashflow or labour.
    p_name : List
        Names of the period you are matching ie cashflow or labour.

    Returns
    -------
    Dataframe 2D
        index = period names you are matching within ie cashflow
        column names = activities ie fertilisers
        This function is used when multiple activities have different period allocation.
        - this func basically just calls the main allocation multiple times and adds the results to one df.
        eg the cost of fert spreading could be in different periods depending what time of yr that fertiliser is applied
    '''
    df = pd.DataFrame()
    for col, start, length in zip(start_df.index, start_df, length_df):
        allocation = period_allocation(p_dates,p_name,start,length)
        df[col]=allocation['allocation']
    df.index=allocation['period']
    return df

#proportion of each feed period that falls in a given date range
#similar to period_allocation that is the proportion of the date range that falls in each period
def range_allocation(period_dates, periods, start, length):
    ''' The proportion of each period that falls in the tested date range.
    Parameters.
    period_dates: the start of the periods - in a DataFrame.
    periods: the period descriptions to be returned in the dataframe.
    start: the date of the beginning of the date range to test.
    length: the length of the date range to test - a timedelta.days object.

    Returns.
    a DataFrame with the period description and the proportion of the period
    '''
    #start empty list to append to
    allocation_period = pd.DataFrame()
    end = start + length
    #check how much of each period falls within the date range
    for i in range(len(periods)-1):
        per_start= period_dates[i].date()    # convert from TimeStamp to datetime
        per_end = period_dates[i + 1].date() #   to allow the calculations
        calc_start = max(per_start,start)       #select the later of the period start or the start of the range
        calc_end = min(per_end,end)             #select earlier of the period end and the end of the range
        allocation=max(0, (calc_end - calc_start) / (per_end - per_start)) #this will be 2d when other pastures are added #days between calc_end and calc_start (0 if end before start) divided by length of the period
        allocation_period=allocation_period.append(pd.DataFrame(data=[allocation]))
    return allocation_period

def range_allocation_np(period_dates, start, length):
    ''' Numpy version - The proportion of each period that falls in the tested date range.

    Parameters.
    period_dates: the start of the periods - in a Numpy array np.datetime64.
    start: the date of the beginning of the date range to test - a numpy array of dates.
    length: the length of the date range to test - an array of timedelta.days object.
       length must be braodcastable into start

    Returns.
    a Numpy array with shape(period_dates, start array).
    Containing the proportion of the respective period for that test date
    '''
    #start empty list to append to
    allocation_period=np.zeros(((len(period_dates),) + start.shape),dtype=np.float64)
    end = start + length
    #check how much of each period falls within the date range
    for i in range(len(period_dates)-1):
        per_start= period_dates[i]
        per_end = period_dates[i + 1]
        calc_start = np.maximum(per_start,start)       #select the later of the period start or the start of the range
        calc_end = np.minimum(per_end,end)             #select earlier of the period end and the end of the range
        allocation_period[i,...] = np.maximum(0, (calc_end - calc_start) / (per_end - per_start)) #days between calc_end and calc_start (0 if end before start) divided by length of the period
    return allocation_period


#the feed period and position in the feed period that a given date range falls
def period_proportion(period_dates, periods, date):
    #check if date falls within period
    period = 0
    proportion = 0
    for i in range(len(periods)-1):
        per_start= period_dates[i]
        per_end = period_dates[i + 1]
        if per_start <= date <= per_end:        #date is within the period
            period = i
            proportion = (date - per_start)/(per_end - per_start)
    return period, proportion

def period_proportion_np(period_dates, date):
    #check if date falls within period
    period = np.zeros(date.shape,dtype='int')
    proportion = np.zeros(date.shape,dtype='float64')
    for i in range(len(period_dates)-1):
        per_start= period_dates[i]
        per_end = period_dates[i + 1]
        if per_start <= date <= per_end:        #date is within the period
            period[...] = i
            # proportion[...] = np.divide(np.subtract(date , per_start),np.subtract(per_end , per_start))
            proportion[...] = (date - per_start) / (per_end - per_start)
    return period, proportion

# #################################################
# # create a numpy by broadcasting dataframes     #
# #################################################

# #this function returns a 2D or 3D numpy array
# #array is created by matrix multiplication of two or three 1D dataframes
# #The dfs passed becomes axis 0, 1 & 2 of the array. The first & third need to be reshaped
# def create_array_from_dfs(df1,df2,df3=''):
#     np1=np.array(df1.to_numpy).reshape(-1,1)
#     np2=np.array(df2.to_numpy)
#     final=np.multiply(np1,np2)
#     if df3:                                         #if there is a 3rd dimension
#         np3=np.array(df3.to_numpy).reshape(1,1,-1)
#         final=np.multiply(final,np3)
#     return final
