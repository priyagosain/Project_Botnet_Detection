# This python code is for a supervised learning mechanism
# where Decision Trees are used to distinguish between the
# different IoT devices that pertains to two botnets ( Bashlite & Mirai).
# 9 classes are to be predicted
# Bashlite Doorbell - Class 1
# Mirai Doorbell - Class 2
# Bashlite Thermostat - Class 3
# Mirai Thermostat - Class 4
# Bashlite Baby Monitor - Class 5
# Mirai Baby Monitor - Class 6
# Bashlite Security Camera - Class 7
# Mirai Security Camera - Class 8
# Bashlite Webcam - Class 9
# This signifies that it is a Multinomial Classification problem.

# Source: http://scikit-learn.org/stable/modules/tree.html

# Import the required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function to import the Dataset
def import_data():
    traffic_data = pd.read_csv("2_Training_Dataset_Bashlite_Mirai.csv")
    return traffic_data

# Function to split the dataset
def split_dataset(traffic_data):
    # Seperating the target variable, y = target, x = data
    x = traffic_data.values[:, 1:7]
    y = traffic_data.values[:, 0]

    # Spliting the dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    return x, y, x_train, x_test, y_train, y_test

# train the model for Decision Trees
def train_using_decision_tree(x_train, y_train):
    # Creating the classifier object
    clf = DecisionTreeClassifier(criterion="entropy", random_state=100)
    # Fitting the training model
    clf.fit(x_train, y_train)
    # printing the classifier made
    return clf

# function to make the predictions
def prediction(x_test, clf_object):
    # Predicton on test with Decision Treee
    y_pred = clf_object.predict(x_test)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n",
          confusion_matrix(y_test, y_pred))

    print("Accuracy of DT (Bashlite and Mirai - 9 classes):\n ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report :\n ",
          classification_report(y_test, y_pred))

# Function to create decision tree graph using Graphviz for visualisation
# There are 9 classes to be classified and hence the 9 labels
# Feature set comprises of Host_Packets_1, Host_Packets_2, HH_Packets_1, HH_Packets_2, HH_Jitter_1, HH_Jitter_2
def graph(clf):
    label_names = ["Doorbell-Bash", "Doorbell-Mirai", "Thermostat-Bash", "Thermostat-Mirai", "BabyMonitor-Bash",
                    "BabyMonitor-Mirai", "Sec-Camera-Bash", "Sec-Camera-Mirai", "Web-cam-Bash" ]
    Feature_names = ["Host_Packets_1", "Host_Packets_2", "HH_Packets_1", "HH_Packets_2", "HH_Jitter_1",
                     "HH_Jitter_2"]

    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=Feature_names,
                         class_names=label_names,
                         filled=True, rounded=True,
                         special_characters=True)
    y_graph = graphviz.Source(dot_data)
    y_graph.render("Bashlite_Mirai_Decision_Rules_Graph")
    return y_graph

# Function to create a Pickle file that contains the Classifier Model
# which can be used at later stages to predict any new set of samples
def load_model(clf):
    pkl_filename_2 = 'classifier_model_2.pkl'
    # Open the file to save as pkl file
    classifier_model_2_pkl = open(pkl_filename_2, 'wb')
    pickle.dump(clf, classifier_model_2_pkl)
    # Close the pickle instances
    classifier_model_2_pkl.close()
    print("Decision Tree classifier: ", clf)
    return classifier_model_2_pkl

# The driver code for all the functions
def main():
    # Building Phase
    data = import_data()
    x, y, x_train, x_test, y_train, y_test = split_dataset(data)
    clf = train_using_decision_tree(x_train, y_train)
    y_pred = prediction(x_test, clf)
    cal_accuracy(y_test, y_pred)
    y_graph = graph(clf)
    classifier_model_pkl = load_model(clf)

# Calling the main function
if __name__ == "__main__":
    main()

