# Importing the necessary librararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Import the dataset
data = pd.read_csv("Fraud_check.csv")
data.head()
data.shape
data.describe()

# Unique values in each column
data.nunique()


# Converting the continuous values of "Taxable.Income" into categorial values of "Risky" and "Good"
frd = pd.cut(data["Taxable.Income"], bins=[0,30000,100000],labels =["Risky","Good"])

# Combining the converted values into the original dataset and removing earlier column
data.insert(1,'Tax_income',frd)
data= data.iloc[:,[0,1,2,4,5,6]]
data.head()

#Data exploration
import seaborn as sns

# Value counts
data.Tax_income.value_counts()

sns.countplot(data["Tax_income"], palette="hls")
# Counts for "Good" taxable income is very high compared to "Risky" counts

sns.countplot(data["Undergrad"], palette="hls")
# Counts for customers with Undergraduation are relatively high compared to the customers without undergraduation

sns.countplot(data["Urban"], palette="hls")
# Population from urban and non_urban is almost same

sns.countplot(data["Marital.Status"], palette="hls")
# Marital status of most of the customers are single and others being Married or Divorced

plt.boxplot(data["City.Population"])
# The city population has a range of 25000 to 200000 with mean popluation being near 110000

plt.boxplot(data["Work.Experience"])
# Work experience of the customers vary from 0 years to 30 years, with mean experience being 15 years

# Obtaining the dummy variables for 'Undergrad','Marital.Status','Urban'
data = pd.get_dummies(data, columns =['Undergrad','Marital.Status','Urban'],drop_first=True)
data.head()

# Checking for any NA values
data.isna().sum()
# No NA values in the dataset

data.shape

# Divivng the data into X and Y i.e. predictors and target
X = data.iloc[:,1:]
Y = data.iloc[:,0]

# Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test= train_test_split(X,Y, test_size=0.2)
Decision tree classifier

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

# Accuracy of the model
Accuracy_entropy = accuracy_score(y_pred,y_test)
Accuracy_entropy

# Using the Decision tree model again but using the "gini" as criterion
model1 = DecisionTreeClassifier(criterion="gini")
model1.fit(x_train, y_train)

# Predicting the target values using the model:
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
    clf = DecisionTreeClassifier(criterion="entropy",max_depth = depth, 
                             random_state = 0)
    clf.fit(x_train, y_train)    
    score = clf.score(x_test, y_test)
    accuracy.append(score)
accuracy

# The maximum accuracy can be obtained with max_depth as 3:
# Changing the max_depth to 3 and building the model
model2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model2.fit(x_train,y_train)

# Predicting the target value using the model
y_pred2= model2.predict(x_test)
y_pred2

# Confusion matrix
confusion_matrix(y_pred2,y_test)

# Visualizing the Decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(model2, filled=True)

# Accuracy
Accuracy_depth3 = accuracy_score(y_pred2,y_test)
Accuracy_depth3

# Improving the accuracy of the model using "bagging" and "boosting" techniques:
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#Bagging
bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(x_train, y_train)

Accuracy_bag =bg.score(x_test, y_test)
Accuracy_bag

# Bagging technique hasn't increased the accuracy of the model much
#Boosting
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=50,learning_rate=1)
ada.fit(x_train,y_train)

Accuracy_boost =ada.score(x_test, y_test)
Accuracy_boost

# Boosting technique hasn't increased the accuracy of the model much
# Tabulating all the results
accuracies = {"Method":pd.Series(["Decisiontree_entropy","Decisiontree_gini","Decisiontree_entropy_depth3","Bagging","Boosting"]),"Accuracy_values":(Accuracy_entropy,Accuracy_gini,Accuracy_depth3,Accuracy_bag,Accuracy_boost)}
table_accuracies = pd.DataFrame(accuracies)
table_accuracies

#Observations
# The DecisionTree model with "entropy" criterion and with max_depth of 3 has given the highest accuracy of 84.1 % amongst all
# Bagging(Bootstrapping) technique has also increased the accuracy of the model to 80.8 %