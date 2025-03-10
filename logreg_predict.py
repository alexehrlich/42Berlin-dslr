from LogisticregressionClassifier import LogisticregressionClassifier
import pandas as pd
from describe import describe

def main():
	df_train = pd.read_csv('./datasets/dataset_train.csv')
	df = df.dropna()
	course_df = df.loc[:, 'Arithmancy':]
	house_df = df.loc[:, 'Hogwarts House'].map({'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3})

	stats = describe(course_df)
	for course in stats.columns:
		course_df.loc[:, course] = (course_df.loc[:, course] - stats.loc['mean', course]) / stats.loc['std', course]

	X = course_df.to_numpy()
	y = house_df.to_numpy().reshape((-1, 1))
	model = LogisticregressionClassifier()
	model.fit(X, y)
	model.save_model('weights.pkl')

if __name__ == '__main__':
	main()