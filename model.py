import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

ds = pd.read_csv("input/heart.csv")

x = ds.drop(["output"], axis=1)
y = ds[["output"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
