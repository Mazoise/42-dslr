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
    model = fl.load(sys.argv[2])
    modelDF = pd.DataFrame(columns=["Index", "Hogwarts House"])
    models = {
        "Gryffindor":MyLr(np.array([[1.0], [1.0]])),
    }
    for i in df["Index"]:
        house = "Gryffindor"
        modelDF = modelDF.append({"Index":i, "Hogwarts House":house}, ignore_index=True)
    modelDF.to_csv("houses.csv", index=False)
except Exception as e:
    print("Error: ", e)