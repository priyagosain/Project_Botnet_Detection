# Import the required packages

import pandas as pd
from sklearn.model_selection import train_test_split

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


# The driver code
def main():
    # Building Phase
    data = import_data()
    x, y, x_train, x_test, y_train, y_test = split_dataset(data)


# Calling the main function
if __name__ == "__main__":
    main()

