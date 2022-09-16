import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLr
import sys
from FileLoader import FileLoader
import math


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
        "Ravenclaw":MyLr(np.array([[1.0], [1.0]]))
    }
    data = {
        'Ravenclaw' : "Charms",
        'Slytherin' : "Divination",
        'Gryffindor' : "Flying"
    }
    for key, lr in models.items():
        lr.theta[0] = modelDF.loc[key,"theta0"]
        lr.theta[1] = modelDF.loc[key,"theta1"]
        lr.bounds = np.array([modelDF.loc[key,"min"], modelDF.loc[key,"max"]])
    for i in df["Index"]:
        pred = {}
        for key, value in data.items():
            pred[key] = float(models[key].predict_1(models[key].minmax_1(df.iloc[i, df.columns.get_loc(value)])))
        if (all(value < 0.8 for value in pred.values())):
            house = "Hufflepuff"
        else:
            house = max(pred, key=pred.get)
        houseDF = houseDF.append({"Index":i, "Hogwarts House":house}, ignore_index=True)
    houseDF.to_csv("houses.csv", index=False)
except Exception as e:
    print("Error: ", e)