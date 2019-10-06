# DisasterResponsePipeline
# Project Overview

This project is part of the Udacity DataScience NanoDegree. Its based on data provided by FigureEight containing messages sent during the Haiti earthquake. Objective of the project is to use machine learning to categorize messages based on the nees the sender expresses

# Contents

- app

| - template

| |- master.html: main page of web app
| |- go.html: classification result page of web app
|- run.py: Flask file that runs app, executes the project

- data
|- disaster_categories.csv : original datased with categories  
|- disaster_messages.csv : original dataset with  messages
|- process_data.py : data processing pipeline that loads categories and messages, cleans them and saves them in an sql database
|- DisasterResponse.db : Database created by process_data.py with clean messages & categories

- models
|- train_classifier.py : machine learning model to classify the data in DisasterResponse.db and exports the final classifier as a pickle file
|- classifier.pkl : FInal classifier output from tryin_classifier.py 

- README.md : file being looked at
