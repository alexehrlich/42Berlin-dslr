import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    truth_df = pd.read_csv('datasets/dataset_truth.csv')
    pred_df = pd.read_csv('houses.csv')

    truth = truth_df['Hogwarts House'].to_numpy()
    pred = pred_df['Hogwarts House'].to_numpy()

    total = len(truth)
    correct = np.array([truth == pred]).sum()
    misclassified = total - correct
    print(f"Correct: {correct}, Misclassified: {misclassified}, Accuracy: {(correct/total)*100}")

    labels = truth_df['Hogwarts House'].unique()

    cm = confusion_matrix(truth, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()

    plt.show()
