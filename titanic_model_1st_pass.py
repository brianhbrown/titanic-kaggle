#First pass at titanic data set to predict survival of passengers, this section will do data visualiztion
#I will run a model with very limited data cleaning, to set baseline results
#Future passes, will attempt data cleaning and stacking

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style('whitegrid')

#Importing test and train datasets
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

def df_inspection(dataframe):
    """
    Takes a data frame in and prints out the analysis info in the console
    :param dataframe: data frame from test set
    :return: print analysis in console
    """
    print(dataframe.head())
    print()
    print(dataframe.info())
    print()
    print(dataframe.columns)
    print()
    print(dataframe.describe())
    print()

def drop_col(dataset, column):
    """
    This function takes in a titanic data set and drops an associated column
    :param dataset: The data frame imported from the test or train set
            column: The string header of the column to be removed
    :return: The data frame minus the column specified by the user
    """
    dataset.drop(column, axis=1, inplace=True)
    return dataset

def acc_report(test, pred):
    """
    Prints the confusion matrix and classification report
    """
    print(confusion_matrix(test, pred))
    print('\n')
    print(classification_report(test, pred))
    print('\n')

def impute_age(cols):
    """
    Takes in the titanic data set and fill in the missing age data
    """
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 30

        else:
            return 25

    else:
        return Age

#check how much data is missing
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title("Missing Data Points")
#plt.show()
#Some age data missing, majority of cabin data missing, 2 instances of embarked missing
#Cabin data does not provide many samples, so I will drop from the data set

drop_col(train, "Cabin")
drop_col(test, "Cabin")
df_inspection(train)

#perform a data analysis on the age data to figure out how to address it
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
plt.xlabel("Age")
plt.title("Passengers by Age")
#plt.show()
sns.boxplot(x = "Survived", y = "Age", data = train, palette = "winter")
#plt.show()
sns.boxplot(x = "Pclass", y = "Age", data = train, palette = "winter")
#plt.show()
sns.boxplot(x = "Sex", y = "Age", data = train, palette = "winter")
#plt.show()

#Age skews noticeably higher if the passenger is class 1
#Plot to see if Class has factor on survival

sns.countplot(x = "Survived", data = train, hue = "Pclass")
#plt.show()

#A lot more passengers died from class 3, we will take this into factor for creating dummy values for Age
#I will determine the mean age for each PClass
mean_by_class = train.groupby("Pclass")["Age"].mean()
print(mean_by_class)
print()
#Pclass 1 = 38
#Pclass 2 = 30
#Pclass 3 = 25

#utilizing the function from the UDEMY data science course, to fill the missing age data
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)

#Drop the passengers with the missing embarked column, there are only 2!
train.dropna(inplace=True)

#Recheck the heatmap for any missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title("Missing Data Points")
#plt.show()
#leaves us with 889 passengers

#Analysis on correlation vs survival
sns.countplot(x = "Survived", data = train, hue = "Sex")
#plt.show()
#A lot more females survived
sns.countplot(x = "Survived", data = train, hue = "Embarked")
#plt.show()
#More survived that embarked from C
sns.countplot(x = "Survived", data = train, hue = "Parch")
#plt.show()
sns.countplot(x = "Survived", data = train, hue = "SibSp")
#plt.show()
#People with no one on board with them tended to die


#Converting categorical data to binary values
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked'],axis=1,inplace=True) #drop categorical columns
train = pd.concat([train,sex,embark],axis=1) #add new numeric columns
df_inspection(train)

#repeat for test data
sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked'],axis=1,inplace=True) #drop categorical columns
test = pd.concat([test,sex_test,embark_test],axis=1) #add new numeric columns
df_inspection(test)


#Running different models, check for highest accuracy
#test train split
from sklearn.model_selection import train_test_split
X = train[["Pclass", "Age", "SibSp", "Parch", "male", "Q", "S"]]
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_final = test[["Pclass", "Age", "SibSp", "Parch", "male", "Q", "S"]]

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_predictions = logmodel.predict(X_test)
log_predictions_all = logmodel.predict(X)
en_log_predictions = logmodel.predict(X_final)
print("Logistic Regression")
print()
acc_report(y_test, log_predictions)
#approx 83%


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

#start with elbow method to find best K value
# error_rate = []
# for i in range(1, 40): #loop through the k neighbors and create an array of error rates
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10,6)) #plot the error rate vs K-value to determine a better K
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()
#K = 11

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
knn_pred_all = knn.predict(X)
en_knn_pred = knn.predict(X_final)
print("KNN Classification, K = 11")
print()
acc_report(y_test, knn_pred)
#approx 81%

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dec_predictions = dtree.predict(X_test)
dec_predictions_all = dtree.predict(X)
en_dec_predictions = dtree.predict(X_final)
print("Decision Tree")
print()
acc_report(y_test, dec_predictions)
#approx 76%

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
forest_predictions = rfc.predict(X_test)
forest_predictions_all = rfc.predict(X)
en_forest_predictions = rfc.predict(X_final)
print("Random Forest")
print()
acc_report(y_test, forest_predictions)
#approx 81%

#SVM
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
SVM_predictions = model.predict(X_test)
SVM_predictions_all = model.predict(X)
en_SVM_predictions = model.predict(X_final)
print("SVM without Grid Search")
print()
acc_report(y_test, SVM_predictions)
#approx 82%

#Grid Search
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
grid.fit(X_train,y_train)
print("SVM with Grid Search")
print()
print(grid.best_params_) #print gridsearch results
print()
print(grid.best_estimator_)
print()
grid_predictions = grid.predict(X_test)
grid_predictions_all = grid.predict(X)
en_grid_predictions = grid.predict(X_final)
acc_report(y_test, grid_predictions)
#approx 82%

#add all the results from the 1st level models to the dataset for a second level prediction

def en_categories(dataset, cat):
    """
    Take data set and add all the 1st level model predictions for the ensembling
    Cat = 1, is train data
    Cat = 2, is test data
    """
    if cat == 1:
        dataset["log predictions"] = log_predictions_all
        dataset["KNN"] = knn_pred_all
        dataset["decision tree"] = dec_predictions_all
        dataset["random forest"] = forest_predictions_all
        dataset["SVM"] = grid_predictions_all
    elif cat == 2:
        dataset["log predictions"] = en_log_predictions
        dataset["KNN"] = en_knn_pred
        dataset["decision tree"] = en_dec_predictions
        dataset["random forest"] = en_forest_predictions
        dataset["SVM"] = en_grid_predictions

en_categories(train, 1)
en_categories(test, 2)

#setting up for the second level model
X = train[["log predictions", "KNN", "decision tree", "random forest", "SVM"]]
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_final = test[["log predictions", "KNN", "decision tree", "random forest", "SVM"]]

#Using a logistic regression for the ensemble
logmodel.fit(X_train,y_train)
log_predictions = logmodel.predict(X_test)
final_predictions = logmodel.predict(X_final) #this is the final result
print("Logistic Regression(Ensemble)")
print()
acc_report(y_test, log_predictions)
#approx 90%, for second level, there is an improvement


#KNN for ensemble
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print("KNN Classification, K = 11(Ensemble)")
print()
acc_report(y_test, knn_pred)
#approx 81%

#Decision tree for ensemble
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dec_predictions = dtree.predict(X_test)
print("Decision Tree(Ensemble)")
print()
acc_report(y_test, dec_predictions)

#Random Forest for ensemble
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
forest_predictions = rfc.predict(X_test)
en_forest_predictions = rfc.predict(X)
print("Random Forest(Ensemble)")
print()
acc_report(y_test, forest_predictions)

#Grid search for ensemble
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
grid.fit(X_train,y_train)
print("SVM with Grid Search(Ensemble)")
print()
print(grid.best_params_) #print gridsearch results
print()
print(grid.best_estimator_)
print()
grid_predictions = grid.predict(X_test)
acc_report(y_test, grid_predictions)

#All results improved to about 90% accuracy


#add the final predictions to the survived column of the test data
test["Survived"] = final_predictions
df_inspection(test)

#remove multiple columns to make data frame submittable.
drop_col(test, "Pclass")
drop_col(test, "Name")
drop_col(test, "Age")
drop_col(test, "SibSp")
drop_col(test, "Parch")
drop_col(test, "Ticket")
drop_col(test, "Fare")
drop_col(test, "male")
drop_col(test, "Q")
drop_col(test, "S")
drop_col(test, "log predictions")
drop_col(test, "KNN")
drop_col(test, "decision tree")
drop_col(test, "random forest")
drop_col(test, "SVM")
df_inspection(test)

#write file to csv for submission!
test.to_csv("Brian_Brown_Submission.csv", index = False)