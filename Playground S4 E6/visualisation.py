import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Playground S4 E6/Data/train.csv")
X = data.iloc[:,1:-1]
corr_matrix = X.corr()

sns.heatmap(corr_matrix, 
        annot = False,
        cmap = "coolwarm",
        annot_kws={"fontsize": 10})
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
plt.show()