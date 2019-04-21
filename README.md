# Linnaeus Classifier
This classifier is designed to accept a string of text containing a disability description from a 526 form and return the proper VA approved classification.

## How to use predictor.py
Run *predictor.py* followed by the string of text you would like to analyze.
-Example: _python predictor.py 'Ringing in my ear'


## How to use predictorBulk.py

Run *predictorBulk* followed by the name of the csv file containing the data.
-Example: _python predictorBulk.py NewData.csv_


## How to retrain the model

1) cd into the *preppingScripts* folder. _cd preppingScripts_

2) Run *dataCleanUp.py* with with the name of the csv containing your dataset as well as the name you'd like to give the cleaned up version of your data. 
Important:Make sure you feed and save files in CSV format 
-Example: _python dataCleanUp.py Large_Training.csv CleanData.csv_

3) Run *modelBuilder.py* with the name of the file you just created in step (1).
-Example: _python modelBuilder.py ../data/CleanData.csv_
This will generate the models and automatically save them.


### Python Version: 3.7