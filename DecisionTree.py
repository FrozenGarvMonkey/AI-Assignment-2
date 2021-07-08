import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Data files
heart = pd.read_csv("Data/heart.csv")

# Splitting Dependant and Independant variables
results = heart["output"]
datas = heart.drop(["output", "oldpeak", "slp"], inplace=False, axis=1)

# Initializing Chart
plt.figure()
plt.subplot(1, 3, 1)
plt.title("Depth - Test Accuracy")
plt.xlim(0, 40)
plt.ylim(0, 1)
plt.xlabel("Depth")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 2)
plt.title("Max Depth - Actual Depth")
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel("Max Depth")
plt.ylabel("Actual Depth")

plt.subplot(1, 3, 3)
plt.title("Depth - Train Accuracy")
plt.xlim(0, 40)
plt.ylim(0, 1)
plt.xlabel("Depth")
plt.ylabel("Accuracy")

markers = ["o", "*", ".", "s", "^"]

for marker in markers:
    test_accuracy = []
    train_accuracy = []
    depth_used = []
    train_data, test_data, train_result, test_result = train_test_split(
        datas, results, test_size=0.2, random_state=True
    )
    for depth in range(1, 41):
        classifier = DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=depth
        )
        classifier.fit(train_data, train_result)
        prediction = classifier.predict(test_data)
        test_accuracy.append(classifier.score(test_data, test_result))
        depth_used.append(classifier.get_depth())
        train_accuracy.append(classifier.score(train_data, train_result))

    plt.subplot(1, 3, 1)
    plt.plot(test_accuracy, marker=marker)
    plt.subplot(1, 3, 2)
    plt.plot(depth_used, marker=marker)
    plt.subplot(1, 3, 3)
    plt.plot(train_accuracy, marker=marker)

plt.show()
