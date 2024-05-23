import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Data/train.csv")

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # Standardises training data.
X_test = scaler.transform(X_test)  # Ensures test data scaled exactly as training data.

# Convert back to data frames so we can add new column with multiplied standardised predictors.
X_test = pd.DataFrame(X_test, columns = X.columns)
X_train = pd.DataFrame(X_train, columns = X.columns)

X_train["new"] = X_train["Watersheds"] * X_train["Landslides"]
X_test["new"] = X_test["Watersheds"] * X_test["Landslides"]

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_pred, y_test)

for i in range(len(model.coef_)):
    print(model.coef_[i], X_train.columns[i])