# Linnaeus Classifier created by Bennett Gebken in order to automatically classify claim contentions
from __future__ import print_function
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
# import the objects used by the vectorizer and classifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
# load the classifier
clf = joblib.load(filename='MasterModel.pkl')
# load the vectorizer
vectorizer = joblib.load(filename='vectorizer.pkl')
#initiate variables
active = True
df = pd.DataFrame(columns=['NARRATIVE'])
output = 0
while active:
    # read the incoming data
    contention = input('Enter your contention (type Q to quit): ')
    if contention == "Q":
        active = False
    #read data
    df = df.append({'NARRATIVE': contention}, ignore_index=True)
    #Classify the contention
    x_test = vectorizer.transform(df['NARRATIVE'])
    y_test_pred = clf.predict(x_test)
    df.loc[:, ('Autocode')] = y_test_pred
    df.head(10)
    y_pred_prob = clf.predict_proba(x_test)
    max_index = y_pred_prob[0].argmax()
    code = clf.classes_[max_index]
    top_positions = y_pred_prob.argmax(axis=1)
    top_probabilities = y_pred_prob[np.arange(len(top_positions)), top_positions]
    df.loc[:, ('Probability')] = top_probabilities
    df.head()
    #Print Results
    print(df.loc[[output]])
    output += 1 #increase counter
#dump a copy of the results for analysis
df.to_excel('results.xlsx')
