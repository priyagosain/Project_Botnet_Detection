######## Project_Botn-et_Detection

####Project Flow

#Dataset_Training_Dataset_Zeus_Alexa which is the first dataset we are working on was loaded. It was initially available in pcap version
file. There were 2 pcap files of Zeus and Alexa which were conerted into csv formats using TcpTrace tool and then merged together.Feature 
extraction tools are used for the data pre-processing to collect the features in suitable format for training the classifier model.

#The file 1_LoadModel.py contains code for loading the classifier model to predict the classes for new set of sample for dataset-1, 
use 1_Testing_dataset_Zeus_Alexa.csv and the file for dataset-2, use 1_Testing_dataset_Bashlite_Mirai.csv file.Furthur analysis were 
done on these two datsets. The pickle file needed to chnages accoridng to the dataset in this file. The testing model was read to predict 
the classes and the observations were printed. The save decision tree model pickle is loaded.

#After this step we need to clasifiy the data. There are three clasifier model files namely classifier_model.pkl, classifier_model.pkl_1,
classifier_model_2.pkl. This contains the different classifiers we used to train the model namely decision trees, LinearSVC and Logistic 
Regression.

#We have used the random forest classifier too. This file is named as RandomForest.py. This python code is for a supervised learning 
mechanism where Random Forest Classifiers are used to distinguish between the different IoT devices that pertains to two botnets
( Bashlite & Mirai).
9 classes are to be predicted
Bashlite Doorbell - Class 1
Mirai Doorbell - Class 2
Bashlite Thermostat - Class 3
Mirai Thermostat - Class 4
Bashlite Baby Monitor - Class 5
Mirai Baby Monitor - Class 6
Bashlite Security Camera - Class 7
Mirai Security Camera - Class 8
Bashlite Webcam - Class 9
This signifies that it is a Multinomial Classification problem.
#Steps
#we loaded the dataset and performed the split function while reading the file. 
#Then we trained the model for random forest classifier.
#After that the predictions were made and accuracy was calculated.
#Then we created a Function to create a Pickle file that contains the Classifier Model which can be used at later stages to predict 
any new set of samples namely def load_model(clf).

#The loaded dataset was analysed and trained. This is a binary classification problem. Zeus_Alexa. py contains the code  for a supervised 
learning mechanism where Decision Trees areused to distinguish between the Zeus (Botnet/Illegitimate) and Alexa (Legitimate).This means 
it is a binary classification problem.
#Steps
#we loaded the dataset and performed the split function while reading the file. 
#Then we trained the model for decision tree classifier.
#After that the predictions were made and accuracy was calculated.
#Then we created a Function to create a Pickle file that contains the Classifier Model which can be used at later stages to predict 
any new set of samples.
# Function  is created to make a decision tree graph using Graphviz for visualisation. There are two classes to be classified and 
hence the labels: Illegitimate and Legitimate. Feature set comprises of Duration, Total Packets, Total Bytes, Load, Rate.
#Another Function is created to make a Pickle file that contains the Classifier Model which can be used at later stages to predict
any new set of samples.

#We used the 2_Training_Dataset_Bashlite_Mirai.csv.

#We ploted the graph after the analysis of our second dataset. the visualization is in the file named as Zeus_Alexa_Decision_Rules_Graph.pdf.

#After that we plotted the ROC curve from the results we got after the analysis from both our datasets.
