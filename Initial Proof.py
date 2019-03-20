
from __future__ import print_function
from pprint import pprint
import numpy as np
import pandas as pd

df = pd.read_csv(r'FullTraining.csv')

# split the data frame into separate test, validation, and training data frames
df_training = df[df['CLASS'] <= 'TRAINING']
df_validation = df[df['CLASS'] == 'VALIDATION']
df_test = df[df['CLASS'] == 'TEST']

print('''Data set sizes:
    test: %s
    validation: %s
    training: %s''' % (len(df_test), len(df_validation), len(df_training)))

# Data has now been imported

from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of the CountVectorizer object
vectorizer = CountVectorizer()

# Use the narratives in training data to create the vocabulary that will
# be represented by  feature vectors. This is remembered by the vectorizer.
vectorizer.fit(df_training['NARRATIVE'])

print('Our vectorizer has defined an input vector with %s elements' % len(vectorizer.vocabulary_))
pprint(vectorizer.vocabulary_)

# Convert the training narratives into their matrix representation.
x_training = vectorizer.transform(df_training['NARRATIVE'])

print('''''''''''')
print(x_training.shape)

n_features = x_training.shape[1]
dense_vector = x_training[0].todense()
print('The vector representing our first training narrative looks like this:', dense_vector)
print('Only %s of the %s elements in this vector are nonzero' % (np.count_nonzero(dense_vector), n_features))

vector = vectorizer.transform(['zipties zone'])
print(vector.todense())

from sklearn.linear_model import LogisticRegression

# y_training contains the codes associated with training narratives
y_training = df_training['CODE']

# create an instance of the LogisticRegression model and set regularization to 1.0
clf = LogisticRegression(C=10)

# fit the model to our training data (ie. calculate the weights)
clf.fit(x_training, y_training)

print(clf.coef_.shape)
print(clf.coef_[0])

# print the codes recognized by the classifier
print(clf.classes_)

# select the index of the code whose weights we want to examine
# index 0 corresponds to code 100 (head not elsewhere classified)
# index 3 corresponds to code 121 (ear, external)
code_index = 3
code = clf.classes_[code_index]

# Retrieve the weights for the specified code_index
code_weights = clf.coef_[code_index]

# find the feature_index with largest weight for this code
feature_index = code_weights.argmax()

# create a dictionary that maps from index, to word
feature_mapper = {v: k for k, v in vectorizer.vocabulary_.items()}

# map that index to the word it represents
word = feature_mapper[feature_index]
print('The word with the heighest weight for code %s is "%s"' % (code, word))
print('It has a weight of:', clf.coef_[code_index, feature_index])

from sklearn.metrics import accuracy_score, f1_score

# Convert the validation narratives to a feature matrix
x_validation = vectorizer.transform(df_validation['NARRATIVE'])

# Generate predicted codes for our validation narratives
y_validation_pred = clf.predict(x_validation)

# Calculate how accurately these match the true codes
y_validation = df_validation['CODE']
accuracy = accuracy_score(y_validation, y_validation_pred)
macro_f1 = f1_score(y_validation, y_validation_pred, average='macro')
print('accuracy = %s' % (accuracy))
print('macro f1 score = %s' % (macro_f1))

clf = LogisticRegression(C=1)
clf.fit(x_training, y_training)

y_training_pred = clf.predict(x_training)
training_accuracy = accuracy_score(y_training, y_training_pred)
print('accuracy on training data is: %s' % training_accuracy)

y_validation_pred = clf.predict(x_validation)
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print('accuracy on validation data is: %s' % validation_accuracy)

vectorizer2 = CountVectorizer(min_df=5, ngram_range=(1,2))
vectorizer2.fit(df_training['NARRATIVE'])
print(len(vectorizer2.vocabulary_))

x_training = vectorizer2.transform(df_training['NARRATIVE'])
clf = LogisticRegression(C=1)
clf.fit(x_training, y_training)
y_training_pred = clf.predict(x_training)
training_accuracy = accuracy_score(y_training, y_training_pred)
print('accuracy on training data is: %s' % training_accuracy)

x_validation = vectorizer2.transform(df_validation['NARRATIVE'])
y_validation_pred = clf.predict(x_validation)
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print('accuracy on validation data is: %s' % validation_accuracy)

from sklearn.externals import joblib

# save the classifier object as LRclf.pkl
joblib.dump(clf, filename='LRclf.pkl')
# save the vectorizer object as vectorizer.pkl
joblib.dump(vectorizer2, filename='vectorizer.pkl')

from sklearn.externals import joblib
# import the objects used by our saved vectorizer and classifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# load the classifier
clf = joblib.load(filename='LRclf.pkl')
# load the vectorizer
vectorizer = joblib.load(filename='vectorizer.pkl')


df = pd.read_csv(r'test.csv')
x_test = vectorizer.transform(df_test['NARRATIVE'])
y_test_pred = clf.predict(x_test)
print(x_test.shape)
print(y_test_pred.shape)
print(y_test_pred)

df_test.loc[:, ('Autocode')] = y_test_pred
df_test.head(10)

# Accesssing the Predicted Probabilities
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
df_test.loc[:, ('Probability')] = top_probabilities
df_test.head()

#Saving the results to file
df_test.to_excel('test_autocoded.xlsx')
