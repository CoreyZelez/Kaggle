import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data/train.csv")

vals = data.iloc[:, 1:-1].sum(axis=1)
y = data.iloc[:, -1]

sum_y = dict()
cnts = dict()

for i in range(len(vals)):
    if vals[i] in sum_y:
        sum_y[vals[i]] += y[i]
        cnts[vals[i]] += 1
    else:
        sum_y[vals[i]] = y[i]
        cnts[vals[i]] = 1

average_y = dict()

for val, sum in sum_y.items():
    average_y[val] = sum / cnts[val]

vals = list(average_y.keys())
avgs = list(average_y.values())
cnts = list(cnts.values())

# Plotting
plt.figure(1)
plt.scatter(vals, avgs)

plt.figure(2)
plt.bar(vals, cnts)

plt.show()
