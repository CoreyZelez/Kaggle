import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Income Prediction/Data/adult.csv")

qualitative_columns = ['workclass', 'education', 'marital.status', 'occupation', 
                       'relationship', 'race', 'sex', 'native.country', 'income']
print(data.columns)

data = pd.concat([data, pd.get_dummies(data, columns = qualitative_columns)], axis = 1)
data = data.drop(columns = qualitative_columns)
data = data.drop("income_<=50K", axis = 1)

data_X = data.iloc[:,:-1]
data_y = data.iloc[:,-1]

train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, train_size = 0.7)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

model = LogisticRegression()

model.fit(train_X, train_y)

y_pred = model.predict(test_X)

# Evaluating the model
accuracy = accuracy_score(test_y, y_pred)
conf_matrix = confusion_matrix(test_y, y_pred)
class_report = classification_report(test_y, y_pred)

print("accuracy\n", accuracy)
print("conf_matrix\n", conf_matrix)
print("class_report\n", class_report)

# Create a heatmap
sns.set_theme(style='white')
plt.figure(figsize=(12, 8))
sns.heatmap(data_X.corr(), annot=False, cmap='coolwarm', fmt="0.6f", 
            xticklabels = data_X.columns, 
            yticklabels = data_X.columns)
plt.title('Covariance Matrix Heatmap')
plt.show()

for coef in model.coef_:
    print(coef)
