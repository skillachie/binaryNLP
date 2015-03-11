from categorize_articles import CategorizeArticles
from binary_features import CategorySeries
from pprint import pprint
import os

'''
Script used to generate various NLP binary features
'''

#TODO call categorize articles with start date, end date and the path for the articles
# Get results of categorizations
# Generate binary features
# Save to file

def text_classif_features(directory):

	# - Get predictions
	cat_articles = CategorizeArticles()
	results = cat_articles.run(directory)
	
	# - Get results from proxy object
	predictions = []
	for result in results:
		predictions.append(result)

	# - Generate binary features
	categories_series = CategorySeries()
	series_result = categories_series.get_category_timeseries(predictions)

	series_result.to_csv('nlp_category_binary_features.csv')
	#quantile_series.to_csv("/home/dvc2106/categorization_migration/nlpCategorization/nlp_binary_features.csv")

if __name__ == '__main__':
	directory_location= os.path.join(os.path.dirname(__file__), 'data')
	abs_dir_location = os.path.abspath(directory_location)
	print abs_dir_location
	text_classif_features(directory_location)
