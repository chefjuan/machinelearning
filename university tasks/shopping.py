import csv
import sys
import time

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.2

dataset = pd.read_csv('shopping.csv')

test = dataset.iloc[:, 17:]
print(test.shape)


def main():
    # Check command-line arguments

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data('shopping.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    dataframe = pd.read_csv(filename)
    le = LabelEncoder()
    # columns=['VisitorType', 'Weekend', 'Revenue']

    label2 = le.fit_transform(dataframe['VisitorType'])
    dataframe.drop("VisitorType", axis=1, inplace=True)
    dataframe['VisitorType'] = label2

    label3 = le.fit_transform(dataframe['Weekend'])
    dataframe.drop("Weekend", axis=1, inplace=True)
    dataframe['Weekend'] = label3

    label4 = le.fit_transform(dataframe['Month'])
    dataframe.drop("Month", axis=1, inplace=True)
    dataframe['VisitorType'] = label4

    label1 = le.fit_transform(dataframe['Revenue'])
    dataframe.drop("Revenue", axis=1, inplace=True)
    dataframe['Revenue'] = label1

    y_dataset = dataframe['Revenue'].copy()
    x_dataframe = dataframe.drop(columns='Revenue')

    x_array = x_dataframe.values.tolist()
    y_array = y_dataset.values.tolist()

    return x_array, y_array


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(evidence, labels)
    return (neigh)


def evaluate(labels, predictions):
    labels_size = len(labels)
    size = labels_size

    negatives = 0
    positives = 0
    true_positives = 0
    true_negatives = 0

    for i in range(size):
        if labels[i] == 0:
            negatives += 1

            if labels[i] == predictions[i]:
                true_negatives += 1

        else:
            positives += 1
            if labels[i] == predictions[i]:
                true_positives += 1

    # sensitivity, specificity = evaluate(y_test, predictions)
    return true_positives / positives, true_negatives / negatives


if __name__ == "__main__":
    main()