import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv('./datasets/dataset_train.csv')

    df = df.drop(labels=['First Name', 'Last Name', 'Index', 'Best Hand'], axis = 1)
    
    print(df.describe())    
    df = df.dropna(axis=0)

    print(df.describe())
    # houses = df['Hogwarts House'].unique()
    # fig, axes = plt.subplots(ncols=len(houses))
    # for i, house in enumerate(houses):
    #     filtered = df[df['Hogwarts House'] == house]
    #     axes[i].hist(filtered['Best Hand'])
    #     axes[i].set_title(house)
    # plt.show()


if __name__ == '__main__':
    main()