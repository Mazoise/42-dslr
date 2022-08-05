import numpy as np
from numbers import Number
import math


class TinyStatistician:

    def __init__(self) -> None:
        pass

    def check_format(self, x, p=0.0):
        assert type(x) == list or type(x) == np.ndarray, "not a list"
        assert len(x) > 0, "empty list"
        tmp = np.array(x, dtype=float)
        if len(np.shape(tmp)) > 1:
            assert (np.shape(tmp)[0] == 1 or np.shape(tmp)[1] == 1,
                    "not a vector")
            tmp = tmp.reshape(np.shape(tmp)[0] * np.shape(tmp)[1])
        assert isinstance(p, Number), "not a number"
        assert 0 <= p <= 100, "percentile out of range"
        return tmp

    def mean(self, x):
        try:
            tmp = self.check_format(x)
            return(sum(i for i in tmp) / len(tmp))
        except Exception as e:
            print("Error : ", e)
            return None

    def median(self, x):
        return(self.percentile(x, 50))

    def quartile(self, x):
        tmp = self.percentile(x, 25)
        return([tmp, self.percentile(x, 75)] if tmp is not None else None)

    def percentile(self, x, p):
        try:
            tmp = self.check_format(x, p)
            x2 = np.sort(tmp)
            return x2[int(len(tmp) * p / 100) if p < 100 else len(tmp) - 1]
        except Exception as e:
            print("Error : ", e)
            return None

    def var(self, x):
        try:
            tmp = self.check_format(x)
            return sum((i - self.mean(tmp)) ** 2 for i in tmp) / len(tmp)
        except Exception as e:
            print("Error : ", e)
            return None

    def std(self, x):
        try:
            tmp = self.check_format(x)
            return round(self.var(tmp) ** 0.5, 4)
        except Exception as e:
            print("Error : ", e)
            return None

    def min(self, x):
        try:
            tmp = self.check_format(x)
            min = math.inf
            for i in tmp:
                if i < min:
                    min = i
            return min
        except Exception as e:
            print("Error : ", e)
            return None

    def max(self, x):
        try:
            tmp = self.check_format(x)
            max = -math.inf
            for i in tmp:
                if i > max:
                    max = i
            return max
        except Exception as e:
            print("Error : ", e)
            return None