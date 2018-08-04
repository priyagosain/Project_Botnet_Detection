# Import the required packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

# Function to import the Dataset
def import_data():
    traffic_data = pd.read_csv("1_Training_dataset_Zeus_Alexa.csv")
    return traffic_data

# Function to split the dataset
def split_dataset(traffic_data):
    # Seperating the target variable, Y = target, X = data
    x = traffic_data.values[:, 1:6]
    y = traffic_data.values[:, 0]

    # Spliting the dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100)
    return x, y, x_train, x_test, y_train, y_test

# train the model for Decision Trees
def train_using_decision_tree(x_train, y_train):
    # Creating the classifier object
    clf = DecisionTreeClassifier(criterion="entropy",
                                   random_state=100)
    # Fitting the training model
    clf.fit(x_train, y_train)
    # printing the classifier made
    print("Classifier model:", clf)
    return clf

# function to make the predictions
def prediction(x_test, clf_object):
    # Predicton on test with Decision Treee
    y_pred = clf_object.predict(x_test)
    print(y_pred)
    return y_pred

# Function create decision tree graph using Graphviz for visualisation
# There are two classes to be classified and hence the labels: Illegitimate and Legitimate
# Feature set comprises of Duration, Total Packets, Total Bytes, Load, Rate
def graph(clf):
    label_names = ["Illegitimate", "Legitimate"]
    Feature_names = ["Duration", "Total_Packets", "Total_Bytes", "Load", "Rate"]

    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=Feature_names,
                         class_names=label_names,
                         filled=True, rounded=True,
                         special_characters=True)
    y_graph = graphviz.Source(dot_data)
    y_graph.render("Zeus_Alexa_Decision_Rules_Graph")
    return y_graph

# The driver code for all the functions
def main():
    # Building Phase
    data = import_data()
    x, y, x_train, x_test, y_train, y_test = split_dataset(data)
    clf = train_using_decision_tree(x_train, y_train)
    y_pred = prediction(x_test, clf)
    y_graph = graph(clf)

# Calling the main function
if __name__ == "__main__":
    main()

