# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Import the data
data = pd.read_csv("Company_Data.csv")
data.head()
data.shape
data.describe()

# Unique values in each column
data.nunique()

# Converting the continuous values of "Sales" into categorial values of "High" and "Low"
sales = pd.cut(data.Sales,bins=[0,8,17],labels=['Low','High'])

# Combining the converted values into the original dataset and removing earlier column
data.insert(1,'sales',sales)
data= data.iloc[:,1:12]
data.head()


#Data Exploration
# Value counts for the output variable
data.sales.value_counts()
import seaborn as sns
sns.countplot(data["sales"], palette="hls")
# Counts for "Low" sales is high compared to "High" counts

sns.countplot(data["ShelveLoc"], palette="hls")
# There are more "Medium" quality shleves in many locations compared to "Good" and "Bad" category shelves

sns.countplot(data["Urban"], palette="hls")
# Most of the stores are in "Urban" locations comapred to "Non-urban"

sns.countplot(data["US"], palette="hls")
# Most of the stores are located in "US" and few others in other countries

plt.boxplot(data["Income"])
# Income of the community ranges from 20 to 120 thousands with mean income of around 68 thoudands

plt.boxplot(data["Population"])
# Population of the region ranges from 10 to 509 thousands with mean population of around 260 thousands

plt.boxplot(data["Age"])
# Average age of population ranges from 25 to 80, with mean age being 53
# Obtaining the dummy variables for 'ShelveLoc','Urban','US'

data1 = pd.get_dummies(data, columns=['ShelveLoc','Urban','US'], drop_first = True)
data1.head()

# Checking for NA values in the dataset
data1.isna().sum()
# There is one NA value in the dataset

data1 = data1.dropna()               # Removing the NA value
data1.shape

# Divivng the data into X and Y i.e. predictors and target
X = data1.iloc[:,1:]
Y = data1.iloc[:,0]

# Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test= train_test_split(X,Y, test_size=0.2)

#Decision tree model
# Building the Decision tree model with "entropy" as criterion
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

model = DecisionTreeClassifier(criterion="entropy")
model.fit(x_train, y_train)

# Predicting the target values using the model
y_pred = model.predict(x_test)
y_pred

# Value counts of predicted values
pd.Series(y_pred).value_counts()

# Buliding confusion matrix
confusion_matrix(y_pred, y_test)
pd.crosstab(y_pred, y_test)              # Cross tabulation can also be used

# Accuracy of the model
Accuracy_entropy = accuracy_score(y_pred,y_test)
Accuracy_entropy

# Using the Decision tree model again but using the "gini" as criterion
model1 = DecisionTreeClassifier(criterion="gini")
model1.fit(x_train, y_train)

# Predicting the target values using the model
y_pred1 = model1.predict(x_test)
y_pred1

# Value counts of predicted values
pd.Series(y_pred1).value_counts()

# Buliding the confusion matrix
confusion_matrix(y_pred1, y_test)

# Visualizing the Decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(model1, filled=True)

# Accuracy of the model
Accuracy_gini = accuracy_score(y_pred1,y_test)
Accuracy_gini

# Checking the accuracy of the model with different max_depth values from 3 to 10
# List of values to try for max_depth:
max_depth_range = list(range(3, 10))                   # List to store the average RMSE for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth = depth, 
                             random_state = 0)
    clf.fit(x_train, y_train)    
    score = clf.score(x_test, y_test)
    accuracy.append(score)
accuracy

# The maximum accuracy can be obtained with max_depth as 7
# Changing the max_depth to 7 and building the model
model2 = DecisionTreeClassifier(criterion="gini", max_depth=7)
model2.fit(x_train,y_train)
DecisionTreeClassifier(max_depth=7)

# Predicting the target value using the model
y_pred2= model2.predict(x_test)
y_pred2

# Confusion matrix
confusion_matrix(y_pred2,y_test)

# Visualizing the Decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(model2, filled=True)

# Accuracy
Accuracy_depth =accuracy_score(y_pred2,y_test)
Accuracy_depth

# Improving the accuracy of the model using "bagging" and "boosting" techniques
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#Bagging
bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(x_train, y_train)
BaggingClassifier(base_estimator=DecisionTreeClassifier(), max_samples=0.5,
                  n_estimators=20)
Accuracy_bag =bg.score(x_test, y_test)
Accuracy_bag

# Bagging technique has increased the accuracy of the model to 83.75% from 77.5%
#Boosting
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)
ada.fit(x_train,y_train)
AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=1,
                   n_estimators=20)
Accuracy_boost =ada.score(x_test, y_test)
Accuracy_boost

# Boosting technique hasn't increased the accuracy of the model much
# Tabulating all the results
accuracies = {"Method":pd.Series(["Decisiontree_entropy","Decisiontree_gini","Decisiontree_depth","Bagging","Boosting"]),"Accuracy_values":(Accuracy_entropy,Accuracy_gini,Accuracy_depth,Accuracy_bag,Accuracy_boost)}
table_accuracies = pd.DataFrame(accuracies)
table_accuracies

Observations
# The DecisionTree model with "gini" criterion and with max_depth of 7 has given the high accuracy of 77.5 % 
# Bagging(Bootstrapping) technique has increased the accuracy of the model to 82.5 % which is the highest of all