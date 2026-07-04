import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/creditcard.csv")

sns.countplot(x='Class', data=df)

plt.title("Fraud vs Genuine")

plt.show()