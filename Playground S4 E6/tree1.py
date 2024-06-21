import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Playground S4 E6/Data/train.csv")
X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

qualitative_variables = ["Course", "Application mode", "Previous qualification", "Nacionality", "Mother's qualification",
                        "Mother's occupation", "Father's qualification", "Father's occupation"]

X = pd.get_dummies(X, columns = qualitative_variables)

threshold = 50
for column in X.columns:
    unique_values = X[column].unique()
    if (len(unique_values) == 2 and 0 in unique_values and 1 in unique_values 
        and X[column].sum() < threshold):
        X.drop(column, axis = 1, inplace = True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

cross_val = KFold(n_splits = 5, shuffle=True)  

best_score = -np.Inf
best = ()
for d in range(8, 13):
    for s in [50, 100, 200, 300, 500]:
        tree = DecisionTreeClassifier(max_depth = d, min_samples_split = s)
        score = cross_val_score(tree, X, y, cv = cross_val).mean()
        if score > best_score:
            best_score = score
            best = (d, s)
        print(d, s, score)
    print()

print(best, best_score)




