from __future__ import print_function
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib

# import the objects used by our saved vectorizer and classifier

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# load the classifier
clf = joblib.load(filename='LRclf.pkl')
# load the vectorizer
vectorizer = joblib.load(filename='vectorizer.pkl')

df = pd.read_csv(r'test.csv')

x_test = vectorizer.transform(df['NARRATIVE'])
y_test_pred = clf.predict(x_test)
print(x_test.shape)
print(y_test_pred.shape)
print(y_test_pred)

df.loc[:, ('Autocode')] = y_test_pred
df.head(10)

#Accessing the Predicted Posibilites
y_pred_prob = clf.predict_proba(x_test)
print('The shape of the pred_prob matrix is: %s' % str(y_pred_prob.shape))
print('The probabilities for the first example are:\n%s' % y_pred_prob[0])
max_index = y_pred_prob[0].argmax()
code = clf.classes_[max_index]
print('The highest probability is at index %s which corresponds to code %s' % (max_index, code))

# get a sequence indicating the position with the highest probability for each row
top_positions = y_pred_prob.argmax(axis=1)
top_probabilities = y_pred_prob[np.arange(len(top_positions)), top_positions]
print(top_probabilities.shape)
df.loc[:, ('Probability')] = top_probabilities
df.head()

df.to_excel('test2_autocoded.xlsx')