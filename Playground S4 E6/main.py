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

test_data = pd.read_csv("Playground S4 E6/Data/test.csv")
X_test = test_data.iloc[:,1:]

train_data = pd.read_csv("Playground S4 E6/Data/train.csv")
X_train = train_data.iloc[:,1:-1]
y_train = train_data.iloc[:,-1]

qualitative_variables = ["Course", "Application mode", "Previous qualification", "Nacionality", "Mother's qualification",
                        "Mother's occupation", "Father's qualification", "Father's occupation"]

X_test = pd.get_dummies(X_test, columns = qualitative_variables)
X_train = pd.get_dummies(X_train, columns = qualitative_variables)

threshold = 50
for column in X_train.columns:
    unique_values = X_train[column].unique()
    if (len(unique_values) == 2 and 0 in unique_values and 1 in unique_values 
        and X_train[column].sum() < threshold):
        X_train = X_train.drop(column, axis = 1)

for column in X_train.columns:
    if column not in X_test.columns:
        X_train.drop(column, axis = 1, inplace = True)

for column in X_test.columns:
    if column not in X_train.columns:
        X_test.drop(column, axis = 1, inplace = True)

X_test = X_test[X_train.columns]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tree = DecisionTreeClassifier(max_depth = 10, min_samples_split = 300)
forest = RandomForestClassifier(max_depth = 11, min_samples_split = 20, n_estimators = 700)
logistic_regression = LogisticRegression()

models = [tree, forest, logistic_regression]
model_weights = [2, 3, 4]

for model in models:
    model.fit(X_train, y_train)

model_preds = []
for model in models:
    model_preds.append(model.predict(X_test))

y_pred = []
for i in range(X_test.shape[0]):
    scores = dict()
    for j in range(len(models)):
        pred = model_preds[j][i]
        if pred in scores:
            scores[pred] += model_weights[j]
        else:
            scores[pred] = model_weights[j]
    max_score = -1
    ensemble_pred = None
    for pred in scores:
        if scores[pred] > max_score:
            max_score = scores[pred]
            ensemble_pred = pred
    y_pred.append(ensemble_pred)

results_df = pd.DataFrame({'id': test_data['id'], 'Target': y_pred})
results_df.to_csv("Playground S4 E6/Data/submission.csv", index = False)



