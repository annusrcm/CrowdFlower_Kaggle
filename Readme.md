This code is for Kaggle's Crowdflower Search Results Relevance challenge

Steps to run the code:

1. Install the requirements: sh setup.sh requirements.txt
2. Preprocess the data : python preprocess.py
3. Train the Keras Sequential Model : python train_keras.py
4. Prediction : python predict.py

NOTE: Change the folder paths in config.py as per your system


In step 1, the dependencies to run the code will be installed. In step 2, train.csv and test.csv will be pre processed
and dumped back into the data folder. In step 3, training will happen and model, tokenizer and label encoder will be
dumped into model folder. In step 4, you can perform prediction on fresh data that is test.csv

