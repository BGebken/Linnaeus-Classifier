import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib


# load the vectorizer
vectorizer = joblib.load(filename='modelsAndTransformations/vectorizer.pkl')
# load the classifier
clf = joblib.load(filename='modelsAndTransformations/LRclf.pkl')

def main(arg=None):
    if len(arg) > 1:
        # Load string and clean it
        text = arg[1]
        text = [text.lower().strip()]

        #Vectorize data
        X = vectorizer.transform(text)
        # predict value
        d = {text[0] :clf.predict(X)[0]}
        # print string and value
        print(d)
    else:
        print('please include the string that needs to be scored')

if __name__ == '__main__':
    main(sys.argv)