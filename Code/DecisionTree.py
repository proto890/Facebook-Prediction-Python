#NOTICE THIS CODE WAS DEVELOPED WITH ASSISTANCE FROM BCU MOODLE RESOURCES AND SKLEARN
#Imports Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from imblearn.over_sampling import SMOTE, ADASYN

#Imports required decision tree libraries
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
import graphviz

#Imports accuracy measuring tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
##Imports a time measurement 
import time
start_time = time.time()
#Finds the dataset file and ignores nil values
dataFrame = pd.read_csv('dataset_Facebook.csv', na_values=['?'])
pd.isnull(dataFrame)
#Sets the feature columns, these being "Likes", "Shares" and "Comments"
features = dataFrame.iloc[:, 15:16:17]
#Sets the target column, which is the media type
target = dataFrame.iloc[:, 1]

"""
IMPORTANT NOTICE
The type of social media post is equivilant to a number
They are the following:
    1 = Photo
    2 = Text Status
    3 = Link
    4 = Video
"""
#Splits the data set into test and training
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=10)

##Starts the classifier 
clf = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=8)

##Fits the testing and training data
clf = clf.fit(features_train, target_train)

#Predicts the featured values
predicted = clf.predict(features_test)
#Calculates the Mean Absolute Error
MAE = mean_absolute_error(target_test, predicted)
#Calculates the confusion matrix
ConF = confusion_matrix(target_test, predicted)

target_names = ['Photo', 'Text', 'Link', 'Video']



##Exports a visual image and creates a seperate file in the directory
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("DecisionTreeVisual")


#Sets the file name and file type
with open( "Poste.dot", 'w' ) as f:
    f = tree.export_graphviz( clf, out_file=f )
##Prints the measures of accuracy and results
print("Accuracy:",metrics.accuracy_score(target_test, predicted))
print("Mean Absolute Error:" , ((MAE)))
print("Confusion Matrix:", (ConF))
print("%s seconds" % (time.time() - start_time))
print(classification_report(target_test, predicted, target_names=target_names))
