#! /usr/bin/env python

from categorize_articles import CategorizeArticles
from binary_features import CategorySeries
from pprint import pprint
import datetime
import os
import sys

'''
Script used to generate various NLP binary features
'''

#TODO create as CLI script that accepts parameters

def text_classif_features(directory,match,start,end,outpath,cpus):
	# - Get predictions
	cat_articles = CategorizeArticles()
	results = cat_articles.run(directory,cpus)
	
	# - Get results from proxy object
	predictions = []
	for result in results:
		predictions.append(result)

	# - Generate binary features
	#categories_series = CategorySeries(aggr_freq='daily')
	categories_series = CategorySeries(aggr_freq='hourly')
	series_result = categories_series.get_category_timeseries(predictions)

	series_result.to_csv(outpath)
	#quantile_series.to_csv("/home/dvc2106/categorization_migration/nlpCategorization/nlp_binary_features.csv")

def main(args):
        match='\d+-\d+-\d+-\d+-\d+-\d+',
        start = None
        end = None
        outpath='nlp_category_binary_features.csv'
        cpus = 0
	directory_location= os.path.join(os.path.dirname(__file__), 'data')
        for arg in args:
                if os.path.isdir(arg):
                        directory_location = arg
                elif arg.find('*') > -1:
                        match = arg
                elif arg.find('.csv') > -1:
                        outpath = arg
                elif arg[0].isdigit() and arg.find('/') > -1:
                        parts = arg.split('_')
                        datex = parts[0].split('/')
                        if len(parts) > 1:
                                timex = parts[1].split(':')
                        elif start:
                                timex = [ 13, 59, 99 ]
                        else:
                                timex = [ 0, 0, 0 ]
                        dt=datetime.datetime(int(datex[0]),int(datex[1]),int(datex[2]),
                                             int(timex[0]),int(timex[1]),int(timex[2]))
                        if start:
                                end = dt
                        else:
                                start = dt
                elif arg[0].isdigit():
                        cpus = int(arg)
        if not start:
                start=datetime.datetime(2009,1,1),
        if not end:
                end=datetime.datetime(2014,12,31),
	abs_dir_location = os.path.abspath(directory_location)
	text_classif_features(directory_location,match,start,end,outpath,cpus)

if __name__ == '__main__':
        main(sys.argv[1:])
