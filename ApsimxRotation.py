'''
APSIMX intergration for rotation inputs
'''

import pandas as pd
import sqlite3




##Build .apsimx file for each rotation phase

##Run apsimx files

###############################################
##access and manipulate output from each run. #
###############################################
# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("../APSIM/rotation.db")
df = pd.read_sql_query("SELECT * from Report2", con)

# Verify that result of SQL query is stored in the dataframe
print(df.head())

# close connection
con.close()

