#NOTICE THIS CODE WAS DEVELOPED WITH ASSISTANCE FROM BCU MOODLE RESOURCES AND SKLEARN
#Imports the required frameworks
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
 #Imports a time measurement
import time
start_time = time.time()

#Finds the dataset and ignores nil values
dataFrame = pd.read_csv('dataset_Facebook.csv', na_values=['?'])
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
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 10)
#Starts the SVC classifier
model = svm.SVC(kernel='linear', gamma=1)
##Fits the data to the model
model.fit(features, target)
#generates a model score
model.score(features, target)
#Generates a predicted value
predicted = model.predict(features_test)
#Calculates the mean absolute error 
MAE = mean_absolute_error(target_test, predicted)
#Calculates the confusion matrix
ConF = confusion_matrix(target_test, predicted)

target_names = ['Photo', 'Text', 'Link', 'Video']

#Funcion used to display the data on a graph
def display_data(features, target):
    plt.scatter(features_test, target_test, color='green')#Scatters data
    plt.plot(features_test, predicted, color='red', lw=2, label='Prediction')#Plots prediction
    #Labels the axis
    plt.title('Social media post attention')
    plt.xlabel('Social Media Response')
    plt.ylabel('Post Type')
    plt.legend()
    plt.show()
    
display_data(features,target)

##Prints the accuracy measurements
print("Accuracy: ", (accuracy_score(target_test, predicted, normalize = True)*100))
print("Mean Absolute Error:" , ((MAE)))
print("Confusion Matrix:", (ConF))
print("%s seconds" % (time.time() - start_time))
print(classification_report(target_test, predicted, target_names=target_names))
