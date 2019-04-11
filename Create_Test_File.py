import pandas as pd
df = pd.read_csv('K:/GitHub/Linnaeus-Classifier/cntntnFile.csv')
df.head()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
del df['CNTNTN_CLSFCN_TXT']
del df['CNTNTN_CLSFCN_ID']
df.rename(columns={'Unnamed: 0': 'INDEX'}, inplace=True)
df['CLMANT_TXT'] = df['CLMANT_TXT'].str.replace(r'\W+', ' ')
df['CNTNTN_CLSFCN_TXT'] = df['CNTNTN_CLSFCN_TXT'].str.replace(r'\W+', ' ')

df_test = df[df['INDEX'] >= 6000000]

print(df.tail())

df_test.to_csv('RAW_TEST.csv', index=False)

