import pandas
import os
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
import numpy 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

os.chdir('C:\\Users\\Farooq\\Documents\\Dani\\ECE 4436\\Lab4')

data = pandas.read_csv('Lab4.csv')

pandas.set_option('display.max_columns', None)
#print(data.head())

features = data[['Time', 'Source', 'Destination', 'Length']]
label = data[['Protocol']]

#print(features.head())
#print(label.head())

features = pandas.get_dummies(features, columns=['Source', 'Destination'])
label = pandas.get_dummies(label)

#print(features.head())
print(label.head())

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42) 

model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)

predictionTree = model1.predict(x_test)
predictionKNN = model2.predict(x_test)

y_test = y_test.to_numpy()

y_testl = numpy.argmax(y_test, axis=-1)
predictionTreel = numpy.argmax(predictionTree, axis=-1) 
predictionKNN1 = numpy.argmax(predictionKNN, axis=-1) 

print(accuracy_score(predictionTree, y_test))
print(accuracy_score(predictionKNN, y_test))

print(confusion_matrix(y_testl, predictionTreel, labels = [0,1,2,3,4,5]))
print("----------------------------------------------------")
print(confusion_matrix(y_testl, predictionKNN1, labels = [0,1,2,3,4,5]))