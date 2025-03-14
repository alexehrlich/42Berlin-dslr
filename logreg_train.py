from LogisticregressionClassifier import LogisticregressionClassifier
import sys
import os
from data_utils import test_train_split, preprocess_data, describe
import pandas as pd
import numpy as np


def split(df, ratio=0.75, save=True, show_hist=False):

    if ratio >= 1.0 or ratio < 0.1:
        print("Waring: Unplausible split ratio. Was set to 0.75 instead.\n")
        ratio = 0.75
    train, val = test_train_split(df, ratio=ratio, show_hist=show_hist)

    house_map = {house: i for i, house in enumerate(train.loc[:, 'Hogwarts House'].unique())}
    print(f"Mapped Houses: {house_map}\n")
    train.loc[:, 'Hogwarts House'] = train.loc[:, 'Hogwarts House'].map(house_map)
    val.loc[:, 'Hogwarts House'] = val.loc[:, 'Hogwarts House'].map(house_map)
    house_map_df = pd.DataFrame({'Hogwarts House': list(house_map.keys())}, index=list(house_map.values()))
    house_map_df.to_csv('house_map.csv', index_label='Mapped Index')
    
    if save:
        if not os.path.exists('datasets/splitted/'):
            os.mkdir('datasets/splitted/')
        train.to_csv('datasets/splitted/train.csv')
        val.to_csv('datasets/splitted/val.csv')

        X_train = train.loc[:, ['Defense Against the Dark Arts', 'Divination', 'Ancient Runes', 'Charms', 'Flying']].to_numpy()
        X_val = val.loc[:, ['Defense Against the Dark Arts', 'Divination', 'Ancient Runes', 'Charms', 'Flying']].to_numpy()
        y_train = train.loc[:, 'Hogwarts House'].to_numpy().reshape(-1, 1)
        y_val = val.loc[:, 'Hogwarts House'].to_numpy().reshape(-1, 1)

    return X_train, X_val, y_train, y_val

def read_input(argv):
    if len(sys.argv) != 2:
        print("Wrong number of arguments. Pass one file path to the training data <csv>.")
        return None
    if not os.path.exists(sys.argv[1]):
        print(f"{sys.argv[1]}: Not found.")
        return None
    else:
        return pd.read_csv(sys.argv[1])

def main():
    
    if (df := read_input(sys.argv)) is None:
        return

    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    normalized_df = preprocess_data(df)

    X_train, X_val, y_train, y_val = split(normalized_df, ratio=0.8, show_hist=False)
    
    model = LogisticregressionClassifier()
    model.fit(X_train, X_val, y_train, y_val, epochs=100, alpha=0.001)
    model.save_model('weights.pkl')

if __name__ == '__main__':
    main()
