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
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class Events(object):

  def __init__(self,category_results,aggr_freq='daily'):
    self.category_results = category_results
    self.aggr_freq = aggr_freq

    articles = self._group_by_category()

    # Removing infrequent words
    #self.vec = TfidfVectorizer(input='filename',
     #       stop_words=stopwords.words('english'),
     #       ngram_range=(1,2),
     #       use_idf=True
     #       )

    
  def _group_by_category(self):
    article_categories = defaultdict(list)
    
    for article in self.category_results:
      article_categories[article['category']].append(article)

    return article_categories    


  def _identify_events(self,articles,event_result,debug=False):

    articles_list = []
    for article in articles:
      articles_list.append(article['file'])

    # Removing infrequent words
    self.vec = TfidfVectorizer(input='filename',
            stop_words=stopwords.words('english'),
            ngram_range=(1,2),
            max_features=500,
            sublinear_tf=True,
            use_idf=True
            )

    X = self.vec.fit_transform(articles_list)
    #print("n_samples: %d, n_features: %d" % X.shape)
    #print idf_vector

    af = AffinityPropagation().fit(X)
    #af = AffinityPropagation(damping=0.7).fit(X)
    #cluster_centers_indices = af.cluster_centers_indices_


    # Cluster ids
    labels = af.labels_

    if debug:
      results = pd.DataFrame(articles_list,labels)
      file_name = event_result['date'].strftime("%Y-%m-%d %H:%M:%S") + '_'+ event_result['category']  + '_' + event_result['location'] + 'clusters.csv'
      results.to_csv(file_name)

    return labels, X


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
      print date, category,location

      event_result = defaultdict(dict)

      event_result['date'] = date
      event_result['category'] = category
      event_result['category'] = category
      event_result['location'] = location

      labels, tfidf_vectors = self._identify_events(articles,event_result,debug)
      num_events = len(labels)

      print num_events
      print tfidf_vectors.shape
      
      # Sanity check
      if(len(labels) != tfidf_vectors.shape[0]):
        raise Exception("Number of labels and tf-idf vectors do not match")
     
 
      event_result['num_events'] = num_events
      event_result['labels'] = labels
      #event_result['tfidf_vectors'] = tfidf_vectors

      articles_by_labels = defaultdict(list)
      #tf_idf by date , category and label
      for label, tfidf_vector in zip(event_result['labels'], tfidf_vectors):

        #print '.....tf...'
        #print tfidf_vector.toarray()

        articles_by_labels[label].extend(tfidf_vector.toarray())  
        #articles_by_labels[label].append(tfidf_vector.toarray())  
     
      # Average  cluster article tf_idfs
      events_tfidf = defaultdict(dict)
      for cluster_label in articles_by_labels:
     
        #print articles_by_labels[cluster_label]
        tfidf_avg = np.mean(articles_by_labels[cluster_label],axis=0)
        events_tfidf[cluster_label] = tfidf_avg
        #event_result['events'] = { cluster_label: {'tfidf_avg': tfidf_avg}}

      event_result['events'] = events_tfidf 
      #TODO average intra-cluster similarity

      #event_result['cluster_center_indices'] = cluster_centers_indices

      result_queue.put(event_result)
   
  #TODO make  flexible for hourly as well 
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
      #target=self.count_events, args=((job_queue,result_queue,True),),)
      target=self.count_events, args=((job_queue,result_queue,True),),)
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

