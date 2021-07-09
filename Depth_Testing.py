import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random

random.seed(42)

# Initializing Data
heart = pd.read_csv("Data/heart.csv")
results = heart["output"]
datas = heart.loc[:, ["age", "cp", "trtbps", "chol", "thalachh"]]

# Initializing Chart
fig1, ax1 = plt.subplots(1)
ax1.set_title("Train Accuracy over Max Depth")
ax1.set_xlim(1, 40)
ax1.set_xlabel("Max Depth")
ax1.set_ylabel("Train Accuracy")

fig2, ax2 = plt.subplots(1)
ax2.set_title("Test Accuracy over Max Depth")
ax2.set_xlim(1, 40)
ax2.set_ylim(0.5, 1)
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Test Accuracy")

fig3, ax3 = plt.subplots(1)
ax3.set_title("Actual Depth over Max Depth")
ax3.set_xlim(1, 40)
ax3.set_ylim(0, 20)
ax3.set_xlabel("Max Depth")
ax3.set_ylabel("Actual Depth")

# Initializing Default values
markers = ["o", "*", "s", "^"]
labels = ["Test 1", "Test 2", "Test 3", "Test 4"]
max_depths = [x for x in range(1, 41)]

# Different Trial Runs
for marker in markers:
    test_accuracy = []
    train_accuracy = []
    depth_used = []

    # New Data Split Every Trial Run
    train_data, test_data, train_result, test_result = train_test_split(
        datas, results, test_size=0.2, random_state=random.randint(1, 20), shuffle=True
    )

    # Test Different max_depth Values (1-40)
    for depth in max_depths:
        # New Classifier with Different max_depth Values
        classifier = DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=depth
        )
        classifier.fit(train_data, train_result)
        prediction = classifier.predict(test_data)

        test_accuracy.append(classifier.score(test_data, test_result))
        depth_used.append(classifier.get_depth())
        train_accuracy.append(classifier.score(train_data, train_result))

    # Plotting Trail Run
    ax1.plot(
        max_depths, train_accuracy, marker=marker, label=labels[markers.index(marker)]
    )
    ax2.plot(
        max_depths, test_accuracy, marker=marker, label=labels[markers.index(marker)]
    )
    ax3.plot(max_depths, depth_used, marker=marker, label=labels[markers.index(marker)])

# Marking max depth value chosen
ax1.scatter(9, train_accuracy[8], s=600, facecolors="none", edgecolors="r")
ax3.scatter(9, depth_used[8], s=600, facecolors="none", edgecolors="r")
fig1.savefig("Train Accuracy over Max Depth.png")
fig2.savefig("Test Accuracy over Max Depth.png")
fig3.savefig("Actual Depth over Max Depth.png")

plt.show()
