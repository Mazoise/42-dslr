from FileLoader import FileLoader
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    df.rename(columns=lambda x: textwrap.shorten(x, width=18, placeholder='...'), inplace=True)
    print(df.columns)
    sns.set(font_scale=0.8, rc={"axes.labelsize":6})
    s = sns.pairplot(data=df.iloc[:, 1:], corner=True, diag_kind="hist", hue="Hogwarts House", palette="rocket", plot_kws=dict(marker="+"), diag_kws=dict(element="step"))
    s.set(yticklabels=[], xticklabels=[])
    plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.06)
    plt.show()
except Exception as e:
    print(e)
