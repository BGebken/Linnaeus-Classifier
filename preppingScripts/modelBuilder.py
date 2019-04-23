import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(sys.argv[1], index_col=0)


# Long strings are usually a sign of missformated data so lets stick to fewer than 300 characters for training.
df = df[(df['CLMANT_TXT'].str.len() < 300)]


# Lets separate data points that dont have an approve label from our dataset.
dfOut = df[(~df['CLMANT_TXT'].isnull()) & (df['newClass'].isnull())]
df = df[(~df['CLMANT_TXT'].isnull()) & (~df['newClass'].isnull())]

print('\n')
print('vectorizing text')
print('\n')

# Initialize a Count Vectorizer with a minimum of 10 appearance of a word for significance, 
# english stopwords and up to 3 word ngrams.
vectorizer = CountVectorizer(min_df=10, ngram_range=(1,3), stop_words='english') 

# This generates our feature space
X = vectorizer.fit_transform(df['CLMANT_TXT'])

#This are the labels we are trying to predict
y = np.array(df['CNTNTN_CLSFCN_ID'])

# Split into a training and testing set.
X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, y, df.index, test_size=0.7, random_state=42)

print('\n')
print('Training model. This may take a while and there might be a few warnings but dont worry, it will work.')
print('\n')


# Initialize a Logistic Regression Model.
clf = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=-1, max_iter=1000)

# Train a model
clf.fit(X_train, y_train)

# save the vectorizer object as vectorizer.pkl
joblib.dump(vectorizer, filename='../modelsAndTransformations/vectorizer.pkl')

# save the classifier object as LRclf.pkl
joblib.dump(clf, filename='../modelsAndTransformations/LRclf.pkl')

# Measure accuracy
y_pred = clf.predict(X_test)

score = str(clf.score(X_test, y_test))

print('our models accuracy is: ' + score)

print('\n')

# Measure presicion, recall, f1 score
print('our models weighted precision, recall and f1score are as follows: ')
print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))

print('\n')

results = pd.DataFrame()
results['label'] = y_test
results['predictedLabel'] = y_pred
results['claims'] = df['CLMANT_TXT'].loc[i_test]
results['correctPred'] = results.apply(lambda x: int(x['label'] == x['predictedLabel']), 1)
results = results[['claims', 'label', 'predictedLabel', 'correctPred']]
results.to_csv('../data/testResults.csv')

print('Done. A copy of the test results has been saved to testResults.csv in the data folder')
