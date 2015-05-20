#! /usr/bin/env python

from categorize_articles import CategorizeArticles
from binary_features import CategorySeries
from article_events import Events
from pprint import pprint
import datetime
import os
import sys
import argparse

'''
Script used to generate various NLP binary features
'''

def text_classif_features(directory,match,start,end,outpath,cpus,aggr_freq,add_noise):

    # - Get predictions
    cat_articles = CategorizeArticles()
    results = cat_articles.run(directory,cpus,match,start,end)
    
    # - Get results from proxy object
    predictions = []
    for result in results:
        predictions.append(result)

    categories_series = CategorySeries(start,end,aggr_freq,add_noise)
    series_result = categories_series.get_category_timeseries(predictions)

    series_result.to_csv('category_features.csv')

    # Get  events 
    events = Events(predictions) 
    event_results = events.run()

    event_counts = []
    for result in event_results:
      event_counts.append(result)


    event_series = CategorySeries(start,end,aggr_freq,add_noise)
    event_results = event_series.get_event_timeseries(event_counts)
    event_results.to_csv('event_features.csv') 
   
    mpd = series_result.join(event_results, how='outer')
    mpd.to_csv(outpath)

def split_date(date_str):
    date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return date_dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match', help='directory pattern match',nargs='?',const='\d+-\d+-\d+-\d+-\d+-\d+', type=str,default='\d+-\d+-\d+-\d+-\d+-\d+')
    parser.add_argument('--start_date', help='start date for series',type=str)
    parser.add_argument('--end_date', help='end date to be used',type=str)
    parser.add_argument('--n_cpus', help='# of cores to  be used',nargs='?',const=0,type=int,default=0)
    parser.add_argument('--binary_features_outpath', help='fully qualified file name to be used',nargs='?',const='nlp_category_binary_features.csv',type=str,default='nlp_category_binary_features.csv')
    parser.add_argument('--aggr_freq', help='specify if you want hourly/daily aggregatoins',nargs='?',const='hourly',type=str,default='hourly')
    parser.add_argument('--add_noise', help='add normal distributed noise to data to fix cases when there are no predicted articles for that topic',nargs='?',const=False,type=bool,default=False)
    parser.add_argument('--data_location', help='the location of the news articles folder')
    args = parser.parse_args()

    directory_location = args.data_location
    if not directory_location:
        directory_location= os.path.join(os.path.dirname(__file__), 'data')
   
    start_date = args.start_date
    if not start_date:
       st = '2009-01-01'
       start_date = split_date(st)
    else:
        start_date = split_date(start_date)

    end_date = args.end_date
    if not end_date:
        ed='2014-12-31'
        end_date = split_date(ed)
    else:
        end_date = split_date(end_date)
        
        
    #for arg in args:
    #    if os.path.isdir(arg):
    #        directory_location = arg
    #    elif arg.find('*') > -1:
    #        match = arg
    #    elif arg.find('.csv') > -1:
    #        outpath = arg
    #    elif arg[0].isdigit() and arg.find('/') > -1:
    #        parts = arg.split('_')
    #        datex = parts[0].split('/')
    #        if len(parts) > 1:
    #            timex = parts[1].split(':')
    #        elif start:
    #            timex = [ 13, 59, 99 ]
    #        else:
    #            timex = [ 0, 0, 0 ]
    #            dt=datetime.datetime(int(datex[0]),int(datex[1]),int(datex[2]),
    #                                        int(timex[0]),int(timex[1]),int(timex[2]))
    #        if start:
    #            end = dt
    #        else:
    #            start = dt
    #    elif arg[0].isdigit():
    #        cpus = int(arg)

    #    if not start:
    #            start=datetime.datetime(2009,1,1),
    #    if not end:
    #            end=datetime.datetime(2014,12,31),
    abs_dir_location = os.path.abspath(directory_location)
    text_classif_features(directory_location,args.match,start_date,end_date,args.binary_features_outpath,args.n_cpus,args.aggr_freq,args.add_noise)

if __name__ == '__main__':
        #main(sys.argv[1:])
        main()
