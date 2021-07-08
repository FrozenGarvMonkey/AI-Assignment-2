import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Data files
heart = pd.read_csv("input/heart.csv")
heart_d = heart.copy()
print(heart_d.shape)
heart_d.drop_duplicates(inplace=True)
heart_d.reset_index(drop=True, inplace=True)
heart_d.shape
print(heart_d.shape)

# Splitting Dependant and Independant variables
results = heart_d["output"]
datas = heart_d.drop(["output", "oldpeak", "slp"], inplace=False, axis=1)

# Initializing Chart
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Depth comparison")
plt.xticks([i for i in range(1, 41)])
plt.xlabel("Depth")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.title("Depth comparison")
plt.xticks([i for i in range(1, 41)])
plt.xlabel("Max Depth")
plt.ylabel("Actual Depth")

markers = ["o", "*", ".", "s", "^"]

for marker in markers:
    accuracy = []
    depth_used = []
    train_data, test_data, train_result, test_result = train_test_split(
        datas, results, test_size=0.2, random_state=25
    )
    for depth in range(1, 41):
        classifier = DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=depth
        )
        classifier.fit(train_data, train_result)
        prediction = classifier.predict(test_data)

        accuracy.append(classifier.score(test_data, test_result))
        depth_used.append(classifier.get_depth())
        print(classifier.get_n_leaves())

    plt.subplot(1, 3, 1)
    plt.plot(accuracy, marker=marker)
    plt.subplot(1, 3, 2)
    plt.plot(depth_used, marker=marker)
    plt.subplot(1, 3, 3)
    plt.plot(, marker=marker)

plt.savefig("graph.png", bbox_inches="tight")
plt.show()
