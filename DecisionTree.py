import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Data files
heart = pd.read_csv("input/heart.csv")

# Splitting Dependant and Independant variables
results = heart["output"]
datas = heart.drop(["output", "oldpeak", "slp"], inplace=False, axis=1)

# Initializing Chart
fig1, ax1 = plt.subplots(1)
ax1.set_title("Depth comparison")
ax1.set_xticks([i for i in range(1, 41)])
ax1.set_xlabel("Depth")
ax1.set_ylabel("Accuracy")

fig2, ax2 = plt.subplots(1)
ax2.set_title("Depth comparison")
ax2.set_xticks([i for i in range(1, 41)])
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Actual Depth")

fig3, ax3 = plt.subplots(1)
ax3.set_title("Depth - Train Accuracy")
ax3.set_xlim(0, 40)
ax3.set_ylim(0, 2)
ax3.set_xlabel("Depth")
ax3.set_ylabel("Accuracy")

markers = ["o", "*", ".", "s", "^"]

for marker in markers:
    accuracy = []
    train_accuracy = []
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
        train_accuracy.append(classifier.score(train_data, train_result))

    ax1.plot(accuracy, marker=marker)
    ax2.plot(depth_used, marker=marker)
    ax3.plot(train_accuracy, marker=marker)

fig1.savefig("graph1.png")
fig2.savefig("graph2.png")
fig3.savefig("graph3.png")

plt.show()
