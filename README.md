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
-Output: **{'ringing in my ear': ['hearing loss', '3140']}**

## How to use predictorBulk.py

Run *predictorBulk* followed by the name of the csv file containing the data.
-Example: **python predictorBulk.py NewData.csv**


## Notes

### Performance:

**Accuracy**: 92%

**Weighted Precision**: 91%

**Weighted Recall**: 92%

**Weighted F1 Score** 91%


### Test

A file named **testResults.csv.** is included in the _data_ folder. It contains the following fields:

*index*: index of test elements in larger set.

*CLMANT_TXT*: The text from the 526 form.

*CNTNTN_CLSFCN_TXT*: The classification assigned to it.

*CNTNTN_CLSFCN_ID*: The ID for the assigned classification.

*newClass*: the class that corresponds to *CNTNTN_CLSFCN_TXT* based on string similarity with the approved list of classifications.

*predictedLabel*: The classification assigned by the model.

*predID*: The ID corresponding to the the predicted classification.

*correctPred*: Boolean for filtering based on whether or not the model made the right selection.


#### Python Version: 3.7


Created by: Bennett Gebken

Repackaged by: Nel Abdiel
