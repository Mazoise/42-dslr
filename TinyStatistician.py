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
        try:
            tmp = self.check_format(x)
            x2 = np.sort(tmp)
            q = (len(x2) + 1) / 2
            return x2[math.floor(q) - 1] * (q - int(q)) + x2[math.ceil(q) - 1] * (1 - q + int(q))
        except Exception as e:
            print("Error : ", e)
            return None

    def quartile1(self, x):
        try:
            tmp = self.check_format(x)
            x2 = np.sort(tmp)
            q = (len(x2) + 3) / 4
            return x2[math.ceil(q) - 1] * (q - int(q)) + x2[math.floor(q) - 1] * (1 - q + int(q))
        except Exception as e:
            print("Error : ", e)
            return None

    def quartile3(self, x):
        try:
            tmp = self.check_format(x)
            x2 = np.sort(tmp)
            q = (3*len(x2) + 1) / 4
            return x2[math.ceil(q) - 1] * (q - int(q)) + x2[math.floor(q) - 1] * (1 - q + int(q))
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