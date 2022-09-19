from FileLoader import FileLoader
import sys
import matplotlib.pyplot as plt
import seaborn as sns

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    courses = df.iloc[:, 6:].columns
    fig, ax = plt.subplots(3, 5)
    for i in range(0, len(courses)):
        sns.histplot(data=df, x=courses[i], ax=ax[i//5][i%5], hue="Hogwarts House", palette="rocket", element="step")
    plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.06)
    plt.show()
except Exception as e:
    print(e)
