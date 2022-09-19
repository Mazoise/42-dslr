import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLr
import sys
from FileLoader import FileLoader
import math

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    Houses = pd.get_dummies(df["Hogwarts House"])
    Houses['RavSlyth'] = Houses['Ravenclaw'] + Houses['Slytherin']
    Houses['GryffSlyth'] = Houses['Gryffindor'] + Houses['Slytherin']
    data = {
        'Ravenclaw' : "Charms",
        'Slytherin' : "Divination",
        'Gryffindor' : "Flying",
        'RavSlyth' : "Astronomy",
        'GryffSlyth' : "Herbology"
    }
    modelDF = pd.DataFrame(columns=["house", "theta0", "theta1", "min", "max"])
    for i in data.keys():
        myLR = MyLr(np.array([[1.0], [1.0]]), 1, 50)
        tmp = pd.concat([Houses[i], df[[data[i]]]], axis=1).dropna()
        house =  np.array(tmp[i], dtype=float).reshape(-1, 1)
        cycles = loss = old_loss = 0
        x = myLR.minmax_(np.array(tmp[data[i]]).reshape(-1, 1))
        while cycles < 100 or old_loss - loss > loss * 0.000001:
            old_loss = loss
            myLR.fit_(x, house)
            if math.isnan(myLR.theta[0]):
                print("Alpha :", myLR.alpha)
                myLR.alpha *= 0.1
                myLR.theta = np.array([[1.0], [1.0]])
                loss = math.inf
            else:
                loss = myLR.loss_(myLR.predict_(x), house)
            cycles += 1
        myLR.plot_(x, house)
        modelDF = modelDF.append({"house":i, "theta0":myLR.theta.squeeze()[0], "theta1":myLR.theta.squeeze()[1], "min":myLR.bounds[0], "max":myLR.bounds[1]}, ignore_index=True)
    modelDF.to_csv("model.csv")
except Exception as e:
    print("Error: ", e)
