import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLr
import sys
from FileLoader import FileLoader

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    Houses = pd.get_dummies(df["Hogwarts House"])
    # Raven = np.array(df[["Charms", "Muggle Studies"]]).reshape(-1, 1)
    # Slyth = np.array(df[["Divination"]]).reshape(-1, 1)
    # Gryff = np.array(df[["History of Magic", "Flying", "Transfiguration"]]).reshape(-1, 1)
    data = {
        'Ravenclaw' : np.array(df[["Charms"]]).reshape(-1, 1),
        'Slytherin' : np.array(df[["Divination"]]).reshape(-1, 1),
        'Gryffindor' : np.array(df[["Flying"]]).reshape(-1, 1)
    }
    for i in ['Gryffindor', 'Ravenclaw', 'Slytherin']:
        myLR = MyLr(np.array([[1.0], [1.0]]), 1, 500000)
        house =  np.array(Houses[i], dtype=float).reshape(-1, 1)
        myLR.fit_(data[i], house)
        print(myLR.theta)
        model = { "thetas": myLR.theta.squeeze(), "bounds": myLR.bounds }
        modelDF = pd.DataFrame(data=model)
        modelDF.to_csv("model.csv")
        myLR.plot_(data[i], house)
except Exception as e:
    print("Error: ", e)
