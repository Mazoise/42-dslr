from FileLoader import FileLoader
import sys
import matplotlib.pyplot as plt
import seaborn as sns

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    courses = df.iloc[:, 6:].columns
    print(courses)
    fig, ax = plt.subplots(3, 4)
    for i in range(0, len(courses)):
        if (i != 1):
            sns.scatterplot(data=df, x=courses[i], y=courses[1], ax=ax[(i - (i>1))//4][(i - (i>1))%4], hue="Hogwarts House", palette="rocket", marker='.')
    plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.06)
    plt.show()
except Exception as e:
    print(e)
