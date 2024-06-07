import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Playground S4 E6/Data/train.csv")

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_lst = []
score_lst = []
for k in range(1, X.shape[1], 5):
    remaining = X.columns
    selected = []
    for i in range(k):
        best_score = -np.inf
        best_predictor = None
        for predictor in remaining:
            temp = selected + [predictor]
            model = LogisticRegression()
            model.fit(X_train[temp], y_train)
            y_pred = model.predict(X_test[temp])
            score = accuracy_score(y_test, y_pred)
            if score > best_score:
                best_score = score
                best_predictor = predictor
        selected.append(best_predictor)
    model = LogisticRegression()
    model.fit(X_train[selected], y_train)
    y_pred = model.predict(X_test[selected])
    k_lst.append(k)
    score_lst.append(accuracy_score(y_test, y_pred))
