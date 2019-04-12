import datetime
import random

import numpy as np
import pandas as pd
import re
df = pd.read_csv('K:/GitHub/Linnaeus-Classifier/cntntnFile.csv')
df.head()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# replace anything in text that’s not a lowercase letter or a number or a space with a space
df['CLMANT_TXT'] = df['CLMANT_TXT'].str.replace(r'\W+', ' ')
df['CNTNTN_CLSFCN_TXT'] = df['CNTNTN_CLSFCN_TXT'].str.replace(r'\W+', ' ')
# remove columns that aren't needed

del df['BEGIN_DT']
del df['MED_IND']
del df['WELL_GRNDED_APLCBL_IND']
del df['JRN_DT']
del df['JRN_STATUS_TYPE_CD']
del df['CNTNTN_TYPE_CD']
del df['NOTFCN_DT']
del df['CMPLTD_DT']
del df['CNTNTN_STATUS_TYPE_CD']
del df['ORIG_SOURCE_TYPE_CD']
del df['DGNSTC_TYPE_CD']

#Add the autocode column
df.loc[:, ('Autocode')] = df['CNTNTN_CLSFCN_ID']
# Add Diagnostic Column
df['Diagnostic_Code'] = df['Autocode']
# add the classification dataframe and make the data types explicit
df_Classification = pd.read_csv(r'Contention_Dictionary.csv')
df['Autocode'] = df['Autocode'].astype(str)
df_Classification['Diagnostic_Code'] = df_Classification['Diagnostic_Code'].astype(str)
# Use the second dataframe to look up the correct response
df['Autocode'] = df['Autocode'].map(df_Classification.set_index('Diagnostic_Code')['Contention_Classification'])

df.rename(columns={'Unnamed: 0': 'INDEX'}, inplace=True)
#remove any rows that is missing values
df = df.dropna()
#drop the extra columns
del df['Autocode']
del df['Diagnostic_Code']

# purge bad rows
df = df[~df.CLMANT_TXT.str.contains("cd_cntntn_pkg")]
df = df[~df.CNTNTN_CLSFCN_ID.str.contains("cd_cntntn_pkg")]
df = df[~df.CNTNTN_CLSFCN_TXT.str.contains("cd_cntntn_pkg")]
df = df.dropna()

print(df.tail(50))
#Save the results
df.to_csv("Large_Training.csv", index=False)

