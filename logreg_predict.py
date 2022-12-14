import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FileLoader import FileLoader
from my_logistic_regression import MyLogisticRegression as MyLr
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def plot_train():
    cm = confusion_matrix(houseDF["Hogwarts House"], df["Hogwarts House"], labels=["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])
    disp.plot()
    plt.title("Accuracy :" + str(accuracy_score(houseDF["Hogwarts House"], df["Hogwarts House"])))
    plt.show()

def define_house(i):
    pred = {}
    for key, value in data.items():
        pred[key] = float(models[key].predict_1(models[key].minmax_1(df.iloc[i, df.columns.get_loc(value)])))
    if (pred['Ravenclaw or Slytherin'] > 0.5):
        if (pred['Gryffindor or Slytherin'] > 0.5):
            house = 'Slytherin'
        else:
            house = 'Ravenclaw'
    else:
        if (pred['Gryffindor or Slytherin'] > 0.5):
            house = 'Gryffindor'
        else:
            house = 'Hufflepuff'
    max_h = max(pred, key=pred.get)
    highs = []
    for key in ['Ravenclaw', 'Slytherin', 'Gryffindor']:
        if (pred[key] > 0.5):
            highs.append(key)
    if (house != max_h and len(highs) == 1):
        house = highs[0]
    return house

try:
    assert len(sys.argv) > 2, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    modelDF = fl.load(sys.argv[2])
    modelDF.set_index("house", inplace=True)
    houseDF = pd.DataFrame(columns=["Index", "Hogwarts House"])
    models = {
        "Gryffindor":MyLr(np.array([[1.0], [1.0]])),
        "Slytherin":MyLr(np.array([[1.0], [1.0]])),
        "Ravenclaw":MyLr(np.array([[1.0], [1.0]])),
        "Ravenclaw or Slytherin":MyLr(np.array([[1.0], [1.0]])),
        "Gryffindor or Slytherin":MyLr(np.array([[1.0], [1.0]]))
    }
    data = {
        'Ravenclaw' : "Charms",
        'Slytherin' : "Divination",
        'Gryffindor' : "Flying",
        'Ravenclaw or Slytherin' : "Astronomy",
        'Gryffindor or Slytherin' : "Herbology"
    }
    for key, lr in models.items():
        lr.theta[0] = modelDF.loc[key,"theta0"]
        lr.theta[1] = modelDF.loc[key,"theta1"]
        lr.bounds = np.array([modelDF.loc[key,"min"], modelDF.loc[key,"max"]])
    for i in df["Index"]:
        house = define_house(i)
        houseDF = houseDF.append({"Index":i, "Hogwarts House":house}, ignore_index=True)
    houseDF.to_csv("houses.csv", index=False)
    if (sys.argv[1] == "datasets/dataset_train.csv"):
        plot_train()
except Exception as e:
    print("Error: ", e)
