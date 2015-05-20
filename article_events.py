from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import datetime
import os
import sys
from pprint import pprint
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import pylab
from itertools import cycle
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool,Manager
import  multiprocessing
from multiprocessing.pool import ThreadPool


class Events(object):

  def __init__(self,category_results,aggr_freq='daily'):
    self.category_results = category_results
    self.aggr_freq = aggr_freq

    articles = self._group_by_category()

    # Removing infrequent words
    self.vec = TfidfVectorizer(input='filename',
            stop_words=stopwords.words('english'),
            sublinear_tf=True,
            ngram_range=(1,2),
            use_idf=True
            )

    
  def _group_by_category(self):
    article_categories = defaultdict(list)
    
    for article in self.category_results:
      article_categories[article['category']].append(article)

    return article_categories    

  def _identify_events(self,articles,debug=False):

    articles_list = []
    for article in articles:
      articles_list.append(article['file'])

    X = self.vec.fit_transform(articles_list)
    print("n_samples: %d, n_features: %d" % X.shape)

    af = AffinityPropagation().fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    if debug:
      results = pd.DataFrame(articles_list,labels)
      file_name = directory + 'clusters.csv'
      results.to_csv(file_name)

    return labels, cluster_centers_indices


  def count_events(self,arg_list):
 
    # - Unpack variables
    job_queue = arg_list[0]
    result_queue = arg_list[1] 
    debug = arg_list[2] 

    while not job_queue.empty():
      job = job_queue.get(block=False)
      articles = job[0]
      category = job[1]
      date = job[2]
      location = job[3]
      print date, category

      event_result = {}
      event_result['date'] = date
      event_result['category'] = category
      event_result['category'] = category
      event_result['location'] = location

      labels,cluster_centers_indices = self._identify_events(articles,debug)
      num_events = len(cluster_centers_indices)
      
      event_result['num_events'] = num_events
      result_queue.put(event_result)
    
  def _aggr_articles_by_date(self,articles):

    articles_by_date = defaultdict(list)
        
    for article in articles:
      date = datetime.datetime.strptime(article['date'], '%Y-%m-%d-%H-%M-%S')
      if self.aggr_freq == 'hourly':
        hour = datetime.time(date.hour)
        short_date = datetime.date(date.year, date.month, date.day)
        date = datetime.datetime.combine(short_date, hour)
      else:
        date = datetime.date(date.year, date.month, date.day)

      articles_by_date[date].append(article)      

    return articles_by_date

  def _aggr_articles_by_location(self,articles):

    location_articles = defaultdict(list)
    for article in articles:
      location_articles[article['location']].append(article)

    return location_articles

  def run(self,cpu_count=None):
    if not cpu_count:
      cpu_count = multiprocessing.cpu_count()
    n_procs = cpu_count -1 if cpu_count > 1 else 1

    mgm = Manager()
    job_queue = mgm.Queue()
    result_queue = mgm.Queue()

    # group articles by category
    #TODO generic function for grouping
    articles = self._group_by_category()

    for category in articles.keys():
      articles_by_date = self._aggr_articles_by_date(articles[category])
      for article_date in articles_by_date:
        articles_by_location = self._aggr_articles_by_location(articles_by_date[article_date])
        for location in articles_by_location:
          #Add articles to job queue
          job_queue.put((articles_by_location[location],category,article_date,location))


    print 'Starting Event Identification...'

    # - Pool
    pool = []
    for i in xrange(n_procs):
      p = multiprocessing.Process(
      target=self.count_events, args=((job_queue,result_queue,False),),)
      p.start()
      pool.append(p)

    for p in pool:
      p.join()

    events_results = []
    while not result_queue.empty():
      #Evaluate performance of extend vs flatten
      events_results.append(result_queue.get(block=False))

    print "Event Identification Complete ..."
    return events_results

