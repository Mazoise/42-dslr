from FileLoader import FileLoader
from TinyStatistician import TinyStatistician
import sys
import pandas as pd
import numpy as np

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
        stats.loc['Count', col] = df[col].shape[0]
        stats.loc['Mean', col] = tinyStat.mean(np.array(df[col]))
        stats.loc['Std', col] = tinyStat.std(np.array(df[col]))
        stats.loc['Min', col] = tinyStat.min(np.array(df[col]))
        stats.loc['25%', col] = tinyStat.percentile(np.array(df[col]), 25)
        stats.loc['50%', col] = tinyStat.percentile(np.array(df[col]), 50)
        stats.loc['75%', col] = tinyStat.percentile(np.array(df[col]), 75)
        stats.loc['Max', col] = tinyStat.max(np.array(df[col]))
    fl.display(stats, stats.shape[0])
except Exception as e:
    print(e)
