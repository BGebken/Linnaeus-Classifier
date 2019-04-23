# Linnaeus Classifier
This classifier is designed to accept a string of text containing a disability description from a 526 form and return the proper VA approved classification.

**IMPORTANT**: The model needs to be trained once before running _predictor.py_ or _predictorBulk.py_ to generate the pickled files for Count Vectorizer and the Logistic Regression model. Simply follow the steps below and things should work smoothly

## How to (re)train the model

1) cd into the *preppingScripts* folder. **cd preppingScripts**

2) Run *dataCleanUp.py* with with the name of the csv containing your dataset as well as the name you'd like to give the cleaned up version of your data. 
Important:Make sure you feed and save files in CSV format 
-Example: **python dataCleanUp.py Large_Training.csv CleanData.csv**

3) Run *modelBuilder.py* with the name of the file you just created in step (1).
-Example: **python modelBuilder.py ../data/CleanData.csv**
This will generate the models and automatically save them.

## How to use predictor.py
Run *predictor.py* followed by the string of text you would like to analyze.
-Example: **python predictor.py 'Ringing in my ear'**


## How to use predictorBulk.py

Run *predictorBulk* followed by the name of the csv file containing the data.
-Example: **python predictorBulk.py NewData.csv**


## Performance:

**Accuracy**: 92%

**Weighted Precision**: 91%

**Weighted Recall**: 92%

**Weighted F1 Score** 91%


### Python Version: 3.7


Created by: Bennett Gebken

Repackaged by: Nel Abdiel
