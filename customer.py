# Importing the required packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz


# Function importing Dataset
def importdata():
    customer = pd.read_excel('table1.xlsx')
    print('Dataset Type:', type(customer))

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
    X1 = balance_data.values[:, 1:5]
    Y1 = balance_data.values[:, 5:]

    # Splitting the dataset into train and test
    # stratified Train-Test split
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X1, Y1, test_size=0.2, random_state=1, stratify=Y1)

    print("Training set:", X_train1)
    print("Test set:", X_test1)
    return X1, Y1, X_train1, X_test1, y_train1, y_test1


def check_depth(X_train, Y_train, X_test, Y_test ):
    # List of values to try for max_depth:
    max_depth_range = list(range(1, 5))
    # List to store the accuracy for each value of max_depth:
    accuracy = []
    for depth in max_depth_range:
        clf = DecisionTreeClassifier(max_depth=depth,
                                     random_state=0)
        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        accuracy.append(score)
    return accuracy, max_depth_range


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
def train_using_entropy(X_train, y_train):
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
    print("Predicted values:", y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

    print("Report : ", classification_report(y_test, y_pred))


# main code
def main(X, Y, X_train, X_test, y_train, y_test):
    # Building Phase

    # test the depth of the decision tree
    accuracy, max_depth_range = check_depth(X_train, y_train, X_test, y_test)
    plt.plot(max_depth_range, accuracy)
    plt.title('Access the Depth of Decision Tree')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.savefig('Depth_Test')

    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = train_using_entropy(X_train, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


def predict_new_client(new_client_data, X, Y):

    clf_gini = train_using_gini(X, Y)
    clf_entropy = train_using_entropy(X, Y)
    print("New Client:",new_client_data)

    print("Results Using Gini Index:")
    # Prediction using gini
    y_pred_gini = prediction(new_client_data, clf_gini)
    # plot_tree(clf_gini,data)

    # plot the Decision Train trained using the Gini Index
    dot_data = tree.export_graphviz(clf_gini, feature_names=data.columns[1:5],
                                    class_names=['Not Buying', 'Buying'], filled=True)
    graph = graphviz.Source(dot_data)
    graph.render('decision_tree')

    return y_pred_gini


# Calling main function
if __name__ == "__main__":
    # import data from a xlsx table
    data = importdata()

    # Stratified Train-Test dataset
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    # Proceed to train and test a decision tree; return accuracy test reports
    main(X, Y, X_train, X_test, y_train, y_test)

    # new client:年龄(50), 收入(Medium),非学生，信用记录 (excellent)
    new_client = [[3,2,0,2]]
    new_client = pd.DataFrame(new_client,columns=[ 'Age', 'Incoming','Student', 'Credit Rating'])

    # Predict the result for new data
    predict_new_client(new_client, X, Y)

