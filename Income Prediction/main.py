import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

data = pd.read_csv("Income Prediction/Data/adult.csv")

qualitative_columns = ['workclass', 'education', 'marital.status', 'occupation', 
                       'relationship', 'race', 'sex', 'native.country', 'income']

data = pd.concat([data, pd.get_dummies(data, columns = qualitative_columns)], axis = 1)
data = data.drop(columns = qualitative_columns)
data = data.drop("income_<=50K", axis = 1)

data_X = data.iloc[:,:-1]
data_y = data.iloc[:,-1]

train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, train_size = 0.7)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

accuracy_scores = []
k_values = []

selected_features = []
remaining_features = list(range(train_X.shape[1]))
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(10, 101, 5)):
    while len(selected_features) < k:
        best_feature = None
        best_score = -np.inf
        for feature in remaining_features:
            features_to_use = selected_features + [feature]
            train_X_subset = train_X[:, features_to_use]
            test_X_subset = test_X[:, features_to_use]
            model = LogisticRegression()
            model.fit(train_X_subset, train_y)
            y_pred = model.predict(test_X_subset)
            score = accuracy_score(test_y, y_pred)
            if score > best_score:
                best_score = score
                best_feature = feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    final_features = selected_features
    model = LogisticRegression()  
    model.fit(train_X[:, final_features] , train_y)
    y_pred = model.predict(test_X[:, final_features])
    k_values.append(k)
    accuracy_scores.append(accuracy_score(test_y, y_pred))

plt.plot(k_values, accuracy_scores)
plt.title("Forward Stepwise Regressions")
plt.xlabel('k Value')
plt.ylabel('Accuracy Score')
plt.show()


