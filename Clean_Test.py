import datetime
import random
import numpy as np
import pandas as pd
import re
df = pd.read_csv('K:/GitHub/Linnaeus-Classifier/RAW_TEST.csv')
# remove columns that aren't needed
df = df.loc[:, ['CLMANT_TXT']]
print(df.head())
# replace anything in text that’s not a lowercase letter or a number or a space with a space
df['CLMANT_TXT'] = df['CLMANT_TXT'].str.replace(r'\W+', ' ')

#remove any rows that is missing values
df = df.dropna()
# purge bad rows
df = df[~df.CLMANT_TXT.str.contains("cd_cntntn_pkg")]
df = df.dropna()

print(df.head())
print(df.tail())
#Save the results
df.to_csv("CLEAN_TEST.csv", index=False)

