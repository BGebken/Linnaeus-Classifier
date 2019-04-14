from __future__ import print_function
#Speed Test
import time
start = time.time()

import numpy as np
import pandas as pd
from sklearn.externals import joblib
# load the classifier
clf = joblib.load(filename='LRclf.pkl')
# load the vectorizer
vectorizer = joblib.load(filename='vectorizer.pkl')
# Read the data
df = pd.read_csv(r'CLEAN_TEST.csv', nrows=1000, converters={'CLMANT_TXT': lambda x: str(x)})
df['CLMANT_TXT'] = df['CLMANT_TXT'].str.replace(r'\W+', ' ')
# Predict Results
x_test = vectorizer.transform(df['CLMANT_TXT'])
y_test_pred = clf.predict(x_test)
#Add the autocode column
df.loc[:, ('Autocode')] = y_test_pred
# Add Diagnostic Column
df['Diagnostic_Code'] = df['Autocode']
# add the classification datafram and make the data types explicit
df_Classification = pd.read_csv(r'Contention_Dictionary.csv')
df['Autocode'] = df['Autocode'].astype(str)
df_Classification['Diagnostic_Code'] = df_Classification['Diagnostic_Code'].astype(str)
# Use the second dataframe to look up the correct response
df['Autocode'] = df['Autocode'].map(df_Classification.set_index('Diagnostic_Code')['Contention_Classification'])
# Accesssing the Predicted Probabilities
y_pred_prob = clf.predict_proba(x_test)
max_index = y_pred_prob[0].argmax()
code = clf.classes_[max_index]
# get a sequence indicating the position with the highest probability for each row
top_positions = y_pred_prob.argmax(axis=1)
top_probabilities = y_pred_prob[np.arange(len(top_positions)), top_positions]
df.loc[:, ('Probability')] = top_probabilities
# Saving the results to file
df.to_excel('Speed_Test_Results.xlsx')
#print time it took
end = time.time()
print(end - start)
