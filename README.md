# DisasterResponsePipeline

## Overview
This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
Three components are included in this project:
1. ETL pipeline - represented by the python script process_data.py
2. ML pipeline - represented by the python script train_classifier.py
3. Flask Web App - represented by the python script run.py including the html-files master.html and go.html
### ETL pipeline
Loads the messages and categories from the csv-files, merges the two datasets, cleans the data and finally stores it in a SQLite database.
### ML pipeline
Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline. Afterwards, it trains and tunes a model using GridSearchCV, outputs the results on the test set and exports the final model as a pickle file.
### Flask Web App
Handles user interaction in order to derive appropriate categories. Furthermore, the app shows some visualization regarding the underlying training set.

## Instructions:
Run the following commands in the project's root directory to set up the database and model.

To run ETL pipeline that cleans data and stores in database python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

To run ML pipeline that trains classifier and saves python train_classifier.py DisasterResponse.db classifier.pkl 

Run the following command in the app's directory to run the web app python run.py

Go to http://0.0.0.0:3001/ to use the web app to query your own message and see some visualizations about the original dataset.

## Acknowledgements
Thanks to Udacity.com and Figure Eight for providing this DataEngineering Use Case.
