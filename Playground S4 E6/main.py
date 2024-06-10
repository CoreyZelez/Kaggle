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

# Add missing columns from training data to validation and test data.
for column in X_train.columns:
    if column not in X_test:
        X_test[column] = 0
# Remove extra columns from test data.
for column in X_test.columns:
    if column not in X_train.columns:
        X_test = X_test.drop(column, axis = 1)   

X_test = X_test[X_train.columns]

X_pre_train, X_validate, y_pre_train, y_validate = train_test_split(X_train, y_train, train_size = 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_pre_train= scaler.transform(X_pre_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)

assert(X_train.shape[1] == X_test.shape[1] == X_validate.shape[1])

tree = DecisionTreeClassifier(max_depth = 9)
forest_low_depth = RandomForestClassifier(n_estimators = 400, max_depth = 4)
forest_high_depth = RandomForestClassifier(n_estimators = 200, max_depth = 8)
logistic_regression = LogisticRegression()
lda = LinearDiscriminantAnalysis()

models = [tree, forest_high_depth, logistic_regression, lda]

ensemble = Ensemble(models)
ensemble.fit(X_pre_train, y_pre_train)
ensemble.score_individual_models(X_validate, y_validate)
ensemble.calc_linear_weights(min_weight = 22, max_weight = 28)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

results_df = pd.DataFrame({'id': test_data['id'], 'Target': y_pred})
results_df.to_csv("Playground S4 E6/Data/submission.csv", index = False)



