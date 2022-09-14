from FileLoader import FileLoader
from TinyStatistician import TinyStatistician
import sys
import pandas as pd
import numpy as np
import statistics

try:
    assert len(sys.argv) > 1, "Input Error: missing argument"
    fl = FileLoader()
    df = fl.load(sys.argv[1])
    df.dropna(axis=1, how='all', inplace=True)
    # df.reset_index(drop=True, inplace=True)
    numerics = ['int16', 'int32', 'int64',
                'float16', 'float32', 'float64']
    numdata = df.select_dtypes(include=numerics).columns
    stats = pd.DataFrame(columns = numdata, index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    tinyStat = TinyStatistician()
    for col in numdata:
        data = np.array(df[col])
        data = data[~np.isnan(data)]
        stats.loc['Count', col] = df[col].shape[0]
        stats.loc['Mean', col] = tinyStat.mean(data)
        stats.loc['Std', col] = tinyStat.std(data)
        stats.loc['Min', col] = tinyStat.min(data)
        stats.loc['25%', col] = tinyStat.quartile1(data)
        stats.loc['50%', col] = tinyStat.median(data)
        stats.loc['75%', col] = tinyStat.quartile3(data)
        stats.loc['Max', col] = tinyStat.max(data)
        # stats.loc['Count2', col] = df[col].shape[0]
        # stats.loc['Mean2', col] = np.mean(data)
        # stats.loc['Std2', col] = np.std(data)
        # stats.loc['Min2', col] = np.min(data)
        # stats.loc['25%2', col] = np.percentile(data, 25)
        # stats.loc['50%2', col] = np.percentile(data, 50)
        # stats.loc['75%2', col] = np.percentile(data, 75)
        # stats.loc['Max2', col] = np.max(data)

    fl.display(stats, stats.shape[0])
except Exception as e:
    print(e)
