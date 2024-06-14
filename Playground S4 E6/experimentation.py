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

data = pd.read_csv("Playground S4 E6/Data/train.csv")
X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

qualitative_variables = ["Course", "Application mode", "Previous qualification", "Nacionality", "Mother's qualification",
                        "Mother's occupation", "Father's qualification", "Father's occupation"]

X = pd.get_dummies(X, columns = qualitative_variables)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

# Add missing columns from training data to validation and test data.
for column in X_train.columns:
    if column not in X_test:
        X_test[column] = 0
# Remove extra columns from test data.
for column in X_test.columns:
    if column not in X_train.columns:
        X_test = X_test.drop(column, axis = 1)   

X_test = X_test[X_train.columns]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_score = -np.Inf
best = ()
for d in range(9, 13):
    for n in range(200, 501, 100):
        for s in range(50, 280, 40):
            tree = DecisionTreeClassifier(max_depth = d, min_samples_split = s)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_score:
                best_score = accuracy
                best = (d, n, s)
            print(d, n, s, accuracy_score(y_test, y_pred))
    print()

print(best, best_score)

