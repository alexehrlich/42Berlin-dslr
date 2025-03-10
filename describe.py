import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def mean(values):
    if len(values) > 0:
        return sum(values)/len(values)

def std(values):
    val = np.array(values)
    mn = mean(val)
    diff_vec_sq = (val - mn) ** 2
    return (sum(diff_vec_sq) / (len(values) - 1)) ** (1/2)

def count(values):
    return len(values)

def extreme_values(values, compare):
    current = values[0]
    for val in values:
        if compare(val, current):
            current = val
    return current

def median(values, is_sorted=False):
    if not is_sorted:
        sorted = values.sort_values()
    else:
        sorted = values
    upper_idx = len(sorted) // 2
    if len(sorted) % 2 == 0:
        return (sorted.iloc[upper_idx] + sorted.iloc[upper_idx - 1]) / 2
    else:
        return sorted.iloc[upper_idx]

def lower_percentile(values):
    sorted = values.sort_values()
    return median(sorted[:(len(values) // 2)], True)

def upper_percentile(values):
    sorted = values.sort_values()
    return median(sorted.iloc[(len(values) // 2):], True)

def describe(df):
    print(df)
    shorted_cols = {}
    for col in df.columns:
        if len(col) > 10:
            shorted_cols[col] = col[:11]
    df.rename(columns=shorted_cols, inplace=True)

    stat_measures = ['count', 'std', 'min', '25%', '50%', '75%', 'max']

    stats = pd.DataFrame(columns=df.columns, index=stat_measures)
    stats.loc['count'] = [count(df[col_name]) for col_name in df.columns]
    stats.loc['mean'] = [mean(df[col_name]) for col_name in df.columns]
    stats.loc['std'] = [std(df[col_name]) for col_name in df.columns]
    stats.loc['min'] = [extreme_values(df[col_name], lambda a, b: a < b) for col_name in df.columns]
    stats.loc['25%'] = [lower_percentile(df[col_name]) for col_name in df.columns]
    stats.loc['50%'] = [median(df[col_name]) for col_name in df.columns]
    stats.loc['75%'] = [upper_percentile(df[col_name]) for col_name in df.columns]
    stats.loc['max'] = [extreme_values(df[col_name], lambda a, b: a > b) for col_name in df.columns]

    print("\t" + "".join(key.rjust(15) for key in df.columns))
    for stat in stat_measures:
        print(f"{stat}\t" + "".join(f"{float(val):15.6f}" for val in stats.loc[stat]))

    return stats

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
    describe(df.loc[:, 'Best Hand':])
    #print(df.loc[:, 'Best Hand':].describe())

if __name__ == '__main__':
    main()