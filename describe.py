import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def main():
    df = pd.read_csv('./datasets/dataset_train.csv')
    df = df.dropna()
  
    df_numeric = df.select_dtypes(include='number')
    
    print(df_numeric.describe())    

    print("\ncount {:.6f}".format(count(df['Arithmancy'])))
    print("mean ", mean(df['Arithmancy']))
    print("std ", std(df['Arithmancy']))
    print("min ", extreme_values(df['Arithmancy'], lambda a, b: a < b))
    print("25% ", lower_percentile(df['Arithmancy']))
    print("50% ", median(df['Arithmancy']))
    print("75% ", upper_percentile(df['Arithmancy']))
    print("max ", extreme_values(df['Arithmancy'], lambda a, b: a > b))
    # houses = df['Hogwarts House'].unique()
    # fig, axes = plt.subplots(ncols=len(houses))
    # for i, house in enumerate(houses):
    #     filtered = df[df['Hogwarts House'] == house]
    #     axes[i].hist(filtered['Best Hand'])
    #     axes[i].set_title(house)
    # plt.show()


if __name__ == '__main__':
    main()