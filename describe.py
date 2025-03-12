import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from data_utils import describe

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments.")
        print("Usage: python3 describe.py <path>.")
        return

    try:
        df = pd.read_csv(sys.argv[1])
    except:
        print("Could not open file.")
        return

    df = df.dropna()
    df.drop(['Index', 'First Name', 'Last Name', 'Birthday'], axis=1, inplace=True)
    df['Best Hand'] = df['Best Hand'].map({'Left':0, 'Right':1})


    #ignore the target column: all rows and all colums starting from Best Hand
    stats = describe(df.loc[:, 'Best Hand':], should_print=True)

if __name__ == '__main__':
    main()