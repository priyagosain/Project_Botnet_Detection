import pickle
import pandas as pd
import numpy as np

traffic_data = pd.read_csv("1_Testing_dataset_Zeus_Alexa.csv", encoding='utf-8')
# Printing the dataswet shape
print("Dataset Length: ", len(traffic_data))
print("Dataset Shape: ", traffic_data.shape)

# Printing the dataset obseravtions
print(traffic_data.head())

# load the model from disk
decision_tree_pkl_filename = 'classifier_model.pkl'
# Loading the saved decision tree model pickle
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)
print("Loaded Decision gini format tree model :: ", decision_tree_model)

predicted_values = decision_tree_model.predict(traffic_data)
pred = np.savetxt('PredictedValues_zeus.csv', predicted_values, delimiter=',')
print("Predicted Values:", predicted_values)