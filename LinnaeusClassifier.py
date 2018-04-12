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
# read the incoming data
df = pd.read_csv(r'test.csv')

# mark the raw data as "narrative"
x_test = vectorizer.transform(df['NARRATIVE'])
y_test_pred = clf.predict(x_test)
# add the automatically generated results
df.loc[:, ('Autocode')] = y_test_pred
df.head(10)
# Accessing the Predictions
y_pred_prob = clf.predict_proba(x_test)
max_index = y_pred_prob[0].argmax()
code = clf.classes_[max_index]
# get a sequence indicating the position with the highest probability for each row
top_positions = y_pred_prob.argmax(axis=1)
top_probabilities = y_pred_prob[np.arange(len(top_positions)), top_positions]
df.loc[:, ('Probability')] = top_probabilities
df.head()

df.to_excel('test2_autocoded.xlsx')