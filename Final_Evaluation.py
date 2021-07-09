from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Initializing Data
heart = pd.read_csv("Data/heart.csv")
results = heart["output"]
datas = heart.loc[:, ["age", "cp", "trtbps", "chol", "thalachh"]]

# Accurary Recording Variables
accuracy = []
t_positive = 0
t_negative = 0
f_positive = 0
f_negative = 0

# Split Data Randomly with random_state Seed
train_data, test_data, train_result, test_result = train_test_split(
    datas, results, test_size=0.2, random_state=5, shuffle=True
)
test_result = test_result.values.tolist()

# 50 Trial runs
for i in range(50):
    # New Classifier and Decision Tree Every Run
    classifier = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=9)
    classifier.fit(train_data, train_result)

    # Accuracy Recoding
    results = classifier.predict(test_data)
    accuracy.append(classifier.score(test_data, test_result))
    for i in range(len(results)):
        if results[i] == 0:
            if test_result[i] == 0:
                t_positive += 1
            else:
                f_positive += 1
        else:
            if test_result[i] == 1:
                t_negative += 1
            else:
                f_negative += 1
average_acc = sum(accuracy) / len(accuracy)

# Plotting and Visualizing Bar Chart
plt.figure()
plt.bar(
    ["True Low Risk", "True High Risk", "False Low Risk", "False High Risk"],
    [t_positive, t_negative, f_positive, f_negative],
)
plt.title("Average Accuracy = {:%}".format(average_acc))
plt.savefig("Result Accuracy.png", bbox_inches="tight")
plt.show()

# Plotting and Visualizing Last Trial Run's Decision Tree
plt.figure(figsize=(50, 50))
plot_tree(classifier, filled=True, rounded=False, fontsize=10)
plt.savefig("Decision Tree.png", bbox_inches="tight")
plt.show()
