import os.path
import pandas as pd
from numpy import *

def CalculateResults(file):
    data = pd.read_table(file, header=None)
    print(data.mean())

if __name__ == "__main__":
    CalculateResults(os.path.join("./result_acc_0.15"))
