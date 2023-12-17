# OIBSIP
from google.colab import drive
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

drive.mount('/content/drive/')
data = pd.read_csv('/content/drive/MyDrive/IRIS/Iris.csv')

print(data.head())

print(data.tail())

data.info()

data.describe()

data.isnull().sum()

data.dtypes

sns.countplot(data, x = 'Species')
plt.xlabel('Species')
plt.ylabel('count')
plt.title('Species')
plt.show()

#Preparing of the Data
#Split the data into features (x) and (y). The features are the measurements and labels are the iris species.

x = data.drop('Species', axis = 1)
y = data['Species']

#Visualization of the Data

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa','Virgincia']
s = [60,60,60]
ax.pie(s, labels = l, autopct = '%2.5f%%')
plt.show()

plt.figure(figsize = (20,20))
sns.lineplot(data)

sns.pairplot(data, hue = 'Species')

X = data[:, 0:4]
Y = data[:, 4]
print(x)
print(y)

#Trainig and testing the model

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from numpy import random
X = random.randint(12, size = (3,4))
print(X)
