import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")

submission = pd.DataFrame({"id": test_data["id"], "FloodProbability": y_pred})
submission.to_csv("Data/submission.csv", index = False, header = True)


