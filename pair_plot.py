from FileLoader import FileLoader
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    assert df['Hogwarts House'].notnull().all(), "Data Error: \"Hogwarts House\" column has no value"
    df.rename(columns=lambda x: textwrap.shorten(x, width=18, placeholder='...'), inplace=True)
    sns.set(font_scale=0.8, rc={"axes.labelsize":6})
    s = sns.pairplot(data=df.iloc[:, 1:], diag_kind="hist", hue="Hogwarts House", palette="rocket", plot_kws=dict(marker="+"), diag_kws=dict(element="step"))
    s.set(yticklabels=[], xticklabels=[])
    plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.06)
    plt.show()
except Exception as e:
    print(e)
