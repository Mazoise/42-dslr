import pandas


class FileLoader:

    def __init__(self) -> None:
        pass

    def load(self, path):
        try:
            ret = pandas.read_csv(path, index_col=None)
            # print("Loading dataset of dimensions",
                #   ret.shape[0], "X", ret.shape[1])
            return ret
        except Exception as e:
            print("Fileload Error :", e)
            return None

    def display(self, df, n):
        try:
            if n >= 0:
                print(df.head(n))
            else:
                print(df.tail(-n))
        except Exception as e:
            print("Error :", e)
            return None
