#This python code is for loading the classifier model
# to predict the classes for new set of sample...
# for dataset-1, use 1_Testing_dataset_Zeus_Alexa.csv file
# for dataset-2, use 1_Testing_dataset_Bashlite_Mirai.csv file

#Also, change the pickle file according to the datasets

# Source: https://pythonprogramming.net/python-pickle-module-save-objects-serialization/

# Import the required packages
import pickle
import pandas as pd
import numpy as np

# Read the Testing file for which classes are to be predicted.
traffic_data = pd.read_csv("1_Testing_dataset_Zeus_Alexa.csv", encoding='utf-8')
# Printing the dataset shape
print("Dataset Length: ", len(traffic_data))
print("Dataset Shape: ", traffic_data.shape)

# Printing the dataset obseravtions
print(traffic_data.head())

# load the model from disk
decision_tree_pkl_filename = 'classifier_model.pkl'
# decision_tree_pkl_filename = 'classifier_model_2.pkl'  # uncomment this line for second dataset and comment above line

# Loading the saved decision tree model pickle
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)
print("Loaded Decision tree model :: ", decision_tree_model)

predicted_values = decision_tree_model.predict(traffic_data)
pred = np.savetxt('PredictedValues_zeus.csv', predicted_values, delimiter=',')
# pred = np.savetxt('PredictedValues_zeus.csv', predicted_values, delimiter=',') # uncomment this line for second
# dataset and comment above line
print("Predicted Values:", predicted_values)