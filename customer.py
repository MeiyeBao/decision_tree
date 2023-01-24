
# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree


# Function importing Dataset
def importdata():
    customer = pd.read_excel('table1.xlsx')

# update Age ("<=30":1,"[31,40]":2, ">40":3)
    customer.replace("<=30", 1, inplace=True)
    customer.replace("[31,40]", 2, inplace=True)
    customer.replace(" [31,40]", 2, inplace=True)
    customer.replace(">40", 3, inplace=True)
    customer.replace(" >40", 3, inplace=True)

# update Incoming(Low:1, Medium:2, High:3)
    customer.replace("high", 3, inplace=True)
    customer.replace("medium", 2, inplace=True)
    customer.replace("low", 1, inplace=True)

# update Student status (No:0, Yes:1)
    customer.replace("no", 0, inplace=True)
    customer.replace("yes", 1, inplace=True)

# update Credit Rating (Fair:1, Excellent:2)
    customer.replace("fair", 1, inplace=True)
    customer.replace("excellent", 2, inplace=True)

    # Printing the dataset shape
    print("Dataset Length: ", len(customer))
    print("Dataset Shape: ", customer.shape)

    # Printing the dataset observations
    print("Dataset: ", customer.head())

    return customer


# Function to split the dataset
def splitdataset(balance_data):
    # Separating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 5:]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    print("Training set:", X_train)
    print("Test set:", X_test)
    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=4)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


def plot_tree(clf_gini,data):
    # Visualize the decision tree
    #fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf_gini, filled=True, feature_names= data.columns)

    return plt.show()


# Function to perform training with entropy.
def tarin_using_entropy(X_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


# Driver code
def main():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = tarin_using_entropy(X_train, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


def predict_new_client(new_client_data):
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X, Y)
    clf_entropy = tarin_using_entropy(X, Y)
    print("New Client:",new_client_data)

    print("Results Using Gini Index:")
    # Prediction using gini
    y_pred_gini = prediction(new_client_data, clf_gini)
    plot_tree(clf_gini,data)

    return y_pred_gini


# Calling main function
if __name__ == "__main__":
    main()

    # new client:年龄(50), 收入(Medium),非学生，信用记录 (excellent)
    new_client = [[3,2,0,2]]
    new_client = pd.DataFrame(new_client,columns=[ 'Age', 'Incoming','Student', 'Credit Rating'])
    predict_new_client(new_client)
