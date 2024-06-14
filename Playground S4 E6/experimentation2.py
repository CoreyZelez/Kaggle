import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ensemble import Ensemble

test_data = pd.read_csv("Playground S4 E6/Data/test.csv")
X_test = test_data.iloc[:,1:]

train_data = pd.read_csv("Playground S4 E6/Data/train.csv")
X_train = train_data.iloc[:,1:-1]
y_train = train_data.iloc[:,-1]

qualitative_variables = ["Course", "Application mode", "Previous qualification", "Nacionality", "Mother's qualification",
                        "Mother's occupation", "Father's qualification", "Father's occupation"]

X_train = pd.get_dummies(X_train, columns = qualitative_variables)
X_test = pd.get_dummies(X_test, columns = qualitative_variables)

# Remove dummy columns which have a low number of observations of the associated categorical value.
threshold = 10
for column in X_train.columns:
    unique_values = X_train[column].unique()
    if (len(unique_values) == 2 and 0 in unique_values and 1 in unique_values 
        and X_train[column].sum() < threshold):
        X_train.drop(column, axis = 1, inplace = True)

# Add missing columns from training data to validation and test data.
for column in X_train.columns:
    if column not in X_test:
        X_test[column] = 0
# Remove extra columns from test data.
for column in X_test.columns:
    if column not in X_train.columns:
        X_test = X_test.drop(column, axis = 1)   

X_test = X_test[X_train.columns]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


for k in range(7, 10):
    for s in range(50, 151, 10):
        tree = DecisionTreeClassifier(max_depth = k, min_samples_split = s)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        print(k, s, accuracy_score(y_test, y_pred))





