from FileLoader import FileLoader
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    courses = df.iloc[:, 6:].columns
    stand = preprocessing.StandardScaler()
    df_scaled = stand.fit_transform(df.iloc[1:, 6:])
    df_scaled = pd.DataFrame(df_scaled, columns=list(courses))
    print(df_scaled.columns)
    fig, ax = plt.subplots(3, 5)
    for i in range(0, len(courses)):
        for j in range(0, len(courses)):
            if j != i:
                sns.scatterplot(data=df_scaled, ax=ax[i//5][i%5], x=courses[j], y=courses[i], palette="rocket", marker='+', )
    plt.show()
except Exception as e:
    print(e)
