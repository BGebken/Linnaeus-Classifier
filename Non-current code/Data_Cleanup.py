import datetime
import random

import numpy as np
import pandas as pd
import re
df = pd.read_csv('K:/GitHub/Model-Builder-Linnaeus-Classifier/cntntnFile.csv')
df.head()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# RE_NOT_ALPHA_NUM = re.compile('[^a-z0-9]')
# replace anything in text that’s not a lowercase letter or a number or a space with a space
df['CLMANT_TXT'] = df['CLMANT_TXT'].str.replace(r'\W+', ' ')
df['CNTNTN_CLSFCN_TXT'] = df['CNTNTN_CLSFCN_TXT'].str.replace(r'\W+', ' ')
#remove columns that aren't needed
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
df.rename(columns={'Unnamed: 0': 'INDEX'}, inplace=True)
#remove any rows that is missing values
df = df.dropna()
print(df.tail(50))
#Save the results
df.to_csv("Large_Training.csv", index=False)

