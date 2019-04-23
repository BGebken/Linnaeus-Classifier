import sys
import numpy as np
import pandas as pd

from sklearn.externals import joblib


# load the vectorizer
vectorizer = joblib.load(filename='/modelsAndTransformations/vectorizer.pkl')
# load the classifier
clf = joblib.load(filename='/modelsAndTransformations/LRclf.pkl')

# Load Dataset
df = pd.read_csv(sys.argv[1])
dfL['CLMANT_TXT'] = dfL.apply(lambda x: x['CLMANT_TXT'].lower().strip(), 1)

#Vectorize data
X = vectorizer.transform(df['CLMANT_TXT'])

# Predict Label
df['predictedLabel'] = clf.predict(X)

#Save file
file_name = '../data/predicted' + sys.argv[1]
df.to_csv(file_name)

print('Done. Results have been saved in: ' + file_name)
