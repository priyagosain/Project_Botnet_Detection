###### Project_Botnet_Detection

##Project Flow

##Problem Statements
The project deals with 2 problem statements. 
Problem-1: To distinguish between the botnet (illegitimate) and legitimate dataset. We have zeus and alexa records and hence this involves binary classification learning.
Problem-2: To distinguish between the IoT's botnet devices. Here, 9 classes are to be predicted.
Bashlite Doorbell - Class 1
Mirai Doorbell - Class 2
Bashlite Thermostat - Class 3
Mirai Thermostat - Class 4
Bashlite Baby Monitor - Class 5
Mirai Baby Monitor - Class 6
Bashlite Security Camera - Class 7
Mirai Security Camera - Class 8
Bashlite Webcam - Class 9 
Hence, it involves multinomial classification learning.

##Datasets
1) 1_Training_Dataset_Zeus_Alexa.csv: Dataset used for creating the training model for problem-1 (Zeus_Alexa.py). There were 2 pcap files of Zeus and Alexa which were converted into csv formats using TcpTrace tool and then were merged together to create the dataset. Feature extraction tools are used for the data pre-processing to collect the features in suitable format for training the classifier model.

2) 2_Training_Dataset_Bashlite_Mirai.csv: Dataset used for creating the training model for problem-2.

3) 1_Testing_dataset_Zeus_Alexa.csv: Dataset used to make the predictions for problem-1.

4) 1_Testing_dataset_Bashlite_Mirai.csv: Dataset used to make the predictions for problem-2.

##Python codes
1) Zeus_Alexa.py (in the master branch): This deals with the problem-1 discussed above. The code is used for following actions: 
--To create the training model for 1_Training_Dataset_Zeus_Alexa.csv. 
--The dataset is split into training and testing subsets. 
--The testing subset created is used to make the predictions on the basis of the training model created.
--Furthermore, accuracy score, classification report and confusion matrix are generated. 
--Digraph for the decision tree model is created (Zeus_Alexa_Decision_Rules_Graph.pdf).
--ROC curve for the botnet class is generated too (ROC_curve.png is the output file)
--Pickle is used to save the training model into a pickle file: classifier_model.pkl.
--After this use LoadPickle.py.
--PredictedValues_zeus.csv file would be generated for the provisioned dataset.

2) Bashlite_Mirai.py (in the master branch): This deals with the problem-2 discussed above using Decision Trees. The code is used for following actions: 
--To create the training model for 2_Training_Dataset_Bashlite_Mirai.csv. 
--The dataset is split into training and testing subsets. 
--The testing subset created is used to make the predictions on the basis of the training model created.
--Furthermore, accuracy score, classification report and confusion matrix are generated. 
--Digraph for the decision tree model is created (Bashlite_Mirai_Decision_Rules_Graph.pdf).
--Pickle is used to save the training model into a pickle file: classifier_model_2.pkl.
--After this use LoadPickle.py.
--PredictedValues_mirai.csv file would be generated for the provisioned dataset.


3) LoadPickle.py (in the LoadModel branch): This deals with the problem-1 discussed above. The code is used for following actions: 
-- for loading the classifier model
--to predict the classes for new set of sample.
--for problem-1, use classifier_model.pkl and 1_Testing_dataset_Zeus_Alexa.csv file
--for problem-2, use classifier_model_2.pkl and 1_Testing_dataset_Bashlite_Mirai.csv file
--Change the output file name where the predicted values are captured.

4) RandomForest.py (in the master branch): This deals with the problem-2 discussed above using Random Forest Classifiers. The code is used for following actions: 
--To create the training model for 2_Training_Dataset_Bashlite_Mirai.csv. 
--The dataset is split into training and testing subsets. 
--The testing subset created is used to make the predictions on the basis of the training model created.
--Furthermore, accuracy score, classification report and confusion matrix are generated. 
--Pickle is used to save the training model into a pickle file: classifier_model_3.pkl.


