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

# Classifier
train_data, test_data, train_result, test_result = train_test_split(
    datas, results, test_size=0.2, random_state=5, shuffle=True
)
classifier = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=7)
classifier.fit(train_data, train_result)
results = classifier.predict(test_data)
accuracy = classifier.score(test_data, test_result)

# Accuracy plotting
test_result = test_result.values.tolist()
t_positive = 0
t_negative = 0
f_positive = 0
f_negative = 0

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

plt.figure()
plt.bar(
    ["True Low Risk", "True High Risk", "False Low Risk", "False High Risk"],
    [t_positive, t_negative, f_positive, f_negative],
)
plt.title("Accuracy = {:%}".format(accuracy))
plt.savefig("Result Accuracy.png", bbox_inches="tight")
plt.show()

# initialising chart
plt.figure(figsize=(50, 50))

# Tree Diagram
plot_tree(classifier, filled=True, rounded=False, fontsize=10)
plt.savefig("Decision Tree.png", bbox_inches="tight")

plt.show()
