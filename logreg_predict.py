from LogisticregressionClassifier import LogisticregressionClassifier
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from data_utils import preprocess_data, describe
import numpy as np

def read_input(argv):
    if len(sys.argv) != 3:
        print("Wrong number of arguments. Pass one file path to the training data <csv>.")
        return None, None
    if not os.path.exists(sys.argv[1]):
        print(f"{sys.argv[1]}: Not found.")
        return None, None
    if not os.path.exists(sys.argv[2]):
        print(f"{sys.argv[2]}: Not found. Run <logreg_train.py> first.")
        return None, None

    try:
        model = LogisticregressionClassifier.load_model(sys.argv[2])
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    return pd.read_csv(sys.argv[1]), model

def main():
    
    df, model = read_input(sys.argv)

    if df is None or model is None:
        return

    df.drop(['Hogwarts House'], axis=1, inplace=True)
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    normalized_df = preprocess_data(df)

    X_test = normalized_df.loc[:, ['Defense Against the Dark Arts', 'Divination', 'Ancient Runes', 'Charms', 'Flying']].to_numpy()

    house_map = pd.read_csv('house_map.csv', index_col='Mapped Index')
    results = pd.DataFrame(columns=['Hogwarts House'])
    for x in X_test:
        prediction = np.argmax(model.predict(x.reshape(-1, 1)))
        house =  house_map.loc[prediction, 'Hogwarts House']
        results.loc[len(results)] = [house]
    results.to_csv('houses.csv', index_label='Index')

if __name__ == '__main__':
    main()
