import matplotlib.pyplot as plt
import pandas as pd
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

def test_train_split(df, ratio=0.8 ,show_hist=False):

    split_idx = int(ratio * len(df.index))
    train = df.iloc[:split_idx, :].copy()
    val = df.iloc[split_idx:, :].copy()

    if show_hist:
        fig, axes = plt.subplots(ncols=2)
        axes = axes.flatten()

        house_order = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        house_to_int = {house: i for i, house in enumerate(house_order)}

        # Convert categorical houses into numerical representation
        df['House Encoded'] = df['Hogwarts House'].map(house_to_int)
        train.loc[:, 'House Encoded'] = train['Hogwarts House'].map(house_to_int)
        val.loc[:, 'House Encoded'] = val['Hogwarts House'].map(house_to_int)

        # Compute relative frequencies
        complete_dist = df['Hogwarts House'].value_counts(normalize=True)
        train_dist = train['Hogwarts House'].value_counts(normalize=True)
        val_dist = val['Hogwarts House'].value_counts(normalize=True)

        bins = range(len(house_order) + 1)  # Define bins for categorical data

        axes[0].hist([df['Hogwarts House'], train['Hogwarts House'], val['Hogwarts House']], 
                 label=['Complete dataset', 'Training set', 'Validation set'], 
                 bins=len(df['Hogwarts House'].unique()), alpha=0.7, edgecolor='black')
        axes[0].set_title('Absolute House Distribution')
        axes[0].legend()

        axes[1].hist([df['House Encoded'], train['House Encoded'], val['House Encoded']], 
             bins=bins, 
             weights=[df['Hogwarts House'].map(lambda x: 1/len(df)).values,  
                      train['Hogwarts House'].map(lambda x: 1/len(train)).values,  
                      val['Hogwarts House'].map(lambda x: 1/len(val)).values], 
             label=['Complete dataset', 'Training set', 'Validation set'], 
             alpha=0.7, edgecolor='black')

        # Set xticks at bin centers
        bin_centers = [i + 0.5 for i in range(len(house_order))]  # Shift ticks to center
        axes[1].set_xticks(bin_centers)       # Place ticks in the middle of each bin
        axes[1].set_xticklabels(house_order)  # Assign Hogwarts house names to ticks
        axes[1].set_title('Normalized House Distribution')
        axes[1].legend()

        plt.legend()
        plt.show()

    return train, val

def normalize(df, stats):
    idx = df.columns.get_loc('Arithmancy')
    courses = df.columns[idx:]
    normalized_df = df.copy()
    for course in courses:
        normalized_df.loc[:, course] = (normalized_df.loc[:, course] - stats.loc['mean', course]) / stats.loc['std', course]
    return normalized_df

def describe(df, should_print=False):
    shorted_cols = {}
    for col in df.columns:
        if len(col) > 10:
            shorted_cols[col] = col[:11]
        else:
            shorted_cols[col] = col

    stat_measures = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    stats = pd.DataFrame(columns=df.columns, index=stat_measures)
    stats.loc['count'] = [count(df[col_name]) for col_name in df.columns]
    stats.loc['mean'] = [mean(df[col_name]) for col_name in df.columns]
    stats.loc['std'] = [std(df[col_name]) for col_name in df.columns]
    stats.loc['min'] = [extreme_values(df[col_name], lambda a, b: a < b) for col_name in df.columns]
    stats.loc['25%'] = [lower_percentile(df[col_name]) for col_name in df.columns]
    stats.loc['50%'] = [median(df[col_name]) for col_name in df.columns]
    stats.loc['75%'] = [upper_percentile(df[col_name]) for col_name in df.columns]
    stats.loc['max'] = [extreme_values(df[col_name], lambda a, b: a > b) for col_name in df.columns]

    if (should_print):
        print("\t" + "".join(course.rjust(15) for course in shorted_cols.values()))
        for stat in stat_measures:
            print(f"{stat}\t" + "".join(f"{float(val):15.6f}" for val in stats.loc[stat]))

    return stats

def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop(['Index', 'First Name', 'Last Name', 'Birthday'], axis=1, inplace=True)
    df['Best Hand'] = df['Best Hand'].map({'Left':0, 'Right':1})
    stats = describe(df.loc[:, 'Arithmancy':])
    normalized_df = normalize(df, stats)

    return normalized_df
    