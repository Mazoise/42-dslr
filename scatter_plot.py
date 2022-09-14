from FileLoader import FileLoader
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    courses = df.iloc[:, 6:].columns
    print(courses)
    for j in range(0, len(courses)):
        fig, ax = plt.subplots(3, 4)
        for i in range(0, len(courses)):
            if (i != j):
                sns.scatterplot(data=df, x=courses[i], y=courses[j], ax=ax[(i - (i>j))//4][(i - (i>j))%4], hue="Hogwarts House", palette="rocket", marker='.')
        plt.show()
except Exception as e:
    print(e)
