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
    # Raven = np.array(df[["Charms", "Muggle Studies"]]).reshape(-1, 1)
    # Slyth = np.array(df[["Divination"]]).reshape(-1, 1)
    # Gryff = np.array(df[["History of Magic", "Flying", "Transfiguration"]]).reshape(-1, 1)
    data = {
        'Ravenclaw' : "Charms",
        'Slytherin' : "Divination",
        'Gryffindor' : "Flying"
    }
    for i in ['Slytherin', 'Gryffindor', 'Ravenclaw']:
        myLR = MyLr(np.array([[1.0], [1.0]]), 1, 5000)
        tmp = pd.concat([Houses[i], df[[data[i]]]], axis=1).dropna()
        house =  np.array(tmp[i], dtype=float).reshape(-1, 1)
        cycles = loss = old_loss = 0
        x = myLR.minmax_(np.array(tmp[data[i]]).reshape(-1, 1))
        # print(x)
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
        print(myLR.theta)
        model = { "thetas": myLR.theta.squeeze(), "bounds": myLR.bounds }
        modelDF = pd.DataFrame(data=model)
        modelDF.to_csv("model.csv")
        myLR.plot_(x, house)
except Exception as e:
    print("Error: ", e)
