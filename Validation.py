from __future__ import print_function
from pprint import pprint
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib

df = pd.read_csv(r'Large_Training.csv', converters={'CNTNTN_CLSFCN_ID': lambda x: str(x)})
df['CLMANT_TXT'] = df['CLMANT_TXT'].str.replace(r'\W+', ' ')
df['CNTNTN_CLSFCN_TXT'] = df['CNTNTN_CLSFCN_TXT'].str.replace(r'\W+', ' ')
df['CNTNTN_CLSFCN_TXT'] = df['CNTNTN_CLSFCN_TXT'].astype(str)

# split the data frame into separate test, validation, and training data frames
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


# split into the 3 data frames
TRAINING, VALIDATION, TEST = train_validate_test_split(df)
print('''Data set sizes:
    TEST: %s
    VALIDATION: %s
    TRAINING: %s''' % (len(TEST), len(VALIDATION), len(TEST)))
# Data has now been imported

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# load the classifier
clf = joblib.load(filename='LRclf.pkl')
# load the vectorizer
vectorizer = joblib.load(filename='vectorizer.pkl')

# Convert the validation narratives to a feature matrix
x_validation = vectorizer.transform(VALIDATION['CLMANT_TXT'])
# Generate predicted codes for our validation narratives
y_validation_pred = clf.predict(x_validation)
# Calculate how accurately these match the true codes
y_validation = VALIDATION['CNTNTN_CLSFCN_ID']
accuracy = accuracy_score(y_validation, y_validation_pred)
print('accuracy = %s' % (accuracy))
