'''
17/4/2021 by Michael Young
Module to combine multiple excel documents into one.
Useful when generating Report.xl in multiple cloud instances.

Limitations:
    1. sheets with multiple tables are not fully supported.
    2. All excel files being combined need to be the same.
'''

import os
import pandas as pd

import ReportFunctions as rep



##gets all report files
files = os.listdir('Output')
excel_files=[]
for file in files:                         # loop through Excel files
    if not file.startswith('~') and file.endswith('.xlsx') and file!='combined_file.xlsx':
        excel_files.append(pd.ExcelFile('Output/{0}'.format(file)))

## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('Output/combined_file.xlsx',engine='xlsxwriter')

##loop through sheets and combine across all excel files.
sheets = excel_files[0].sheet_names
df_settings = excel_files[0].parse(sheet_name = 'df_settings', header=[0], index_col=[0])
for sheet in sheets:
    if sheet != 'df_settings':
        df_total = pd.DataFrame()
        l_index = list(range(df_settings.loc[sheet, 'index'])) #gets the index columns
        l_header = list(range(df_settings.loc[sheet, 'cols'])) #gets the header rows
        for report in excel_files:
            df = report.parse(sheet_name = sheet, header=l_header, index_col=l_index)
            df_total = df_total.append(df)
        ###write to new excel file using custom function in ReportFunctions.py
        rep.f_df2xl(writer, df_total, sheet, option=1)
writer.close()