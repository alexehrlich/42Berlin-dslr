from describe import describe
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import preprocess_data

def main():
	df = pd.read_csv('./datasets/dataset_train.csv')
	df = preprocess_data(df)
	course_df = df.loc[:, 'Best Hand':]
	house = df.loc[:, 'Hogwarts House']
	houses_colors = {
		'Ravenclaw': 'red',
		'Slytherin': 'blue',
		'Gryffindor': 'green',
		'Hufflepuff': 'orange'
	}

	grouped = course_df.groupby(house)

	fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(15, 6))
	axes = axes.flatten()

	legend_han = []
	for i, course in enumerate(course_df.columns):
		for house, color in houses_colors.items():
			_, _, patches = axes[i].hist(grouped.get_group(house)[course], color=color, alpha=0.5, label=house, bins=20)
			axes[i].set_title(course)
			if i == 0:
				legend_han.append(patches[0])
	fig.legend(legend_han, houses_colors.keys(), loc='lower right')
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()