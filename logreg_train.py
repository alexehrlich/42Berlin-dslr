from LogisticregressionClassifier import LogisticregressionClassifier
import pandas as pd
from describe import describe


def main():
	train = pd.read_csv('./datasets/splitted/train.csv')
	val = pd.read_csv('./datasets/splitted/val.csv')

	X_train = course_df.to_numpy()
	y = house_df.to_numpy().reshape((-1, 1))
	model = LogisticregressionClassifier()
	model.fit(X, y)
	model.save_model('weights.pkl')

if __name__ == '__main__':
	main()