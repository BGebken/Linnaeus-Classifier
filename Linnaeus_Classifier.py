from __future__ import print_function
from pprint import pprint
import numpy as np
import pandas as pd
# something something
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
# import the objects used by saved vectorizer and classifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
# load the classifier
clf = joblib.load(filename='LRclf.pkl')
# load the vectorizer
vectorizer = joblib.load(filename='vectorizer.pkl')
# Read the data
df = pd.read_csv(r'RAW_TEST.csv', nrows=20000, converters={'CLMANT_TXT': lambda x: str(x)})
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
df.head()
# Saving the results to file
df.to_excel('Results.xlsx')