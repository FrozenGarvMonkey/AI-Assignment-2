import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Data files
heart = pd.read_csv("Data/heart.csv")

# Splitting Dependant and Independant variables
results = heart["output"]
datas = heart.drop(["output", "oldpeak", "slp"], inplace=False, axis=1)

# initialising chart
plt.figure(figsize=(50, 15))

# Classifier
train_data, test_data, train_result, test_result = train_test_split(
    datas, results, test_size=0.2, random_state=True
)

classifier = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=7)
classifier.fit(train_data, train_result)

plot_tree(classifier, filled=True, rounded=False, fontsize=7)
plt.savefig("Decision Tree.png", bbox_inches="tight")

plt.show()
