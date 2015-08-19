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
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity


class Events(object):

  def __init__(self,category_results,aggr_freq='daily'):
    self.category_results = category_results
    self.aggr_freq = aggr_freq

    articles = self._group_by_category()

    # Removing infrequent words
    self.vec = TfidfVectorizer(input='filename',
                stop_words=stopwords.words('english'),
                ngram_range=(1,2),
                max_features=100,
                use_idf=True
              )


    
  def _group_by_category(self):
    article_categories = defaultdict(list)
    
    for article in self.category_results:
      article_categories[article['category']].append(article)

    return article_categories    


  def _identify_events(self,articles,event_result,debug=False):


    articles_list = []
    for article in articles:
      articles_list.append(article['file'])


    X = self.vec.fit_transform(articles_list)
    #words = self.vec.get_feature_names()
    #print("n_samples: %d, n_features: %d" % X.shape)
    #print idf_vector

    labels = AffinityPropagation(max_iter=4000,damping=0.95,convergence_iter=400).fit_predict(X)

    #Aggregate clustering results
    articles_by_labels,articles_path = self._aggr_by_labels(labels,X,articles_list)

    #Updated articles
    labels = []
    articles_list = []
    new_X = []

    for label in articles_path:
      labels.extend([label] * len(articles_path[label]))
      articles_list.extend(articles_path[label])
      new_X.extend(articles_by_labels[label])


   # print labels
    #print "***"
    if debug:
      results = pd.DataFrame(articles_list,labels)
      file_name = event_result['date'].strftime("%Y-%m-%d %H:%M:%S") + '_'+ event_result['category']  + '_' + event_result['location'] + 'clusters.csv'
      results.to_csv(file_name)

      # Write to SQL Lite
      event_entries = []
      for article_path, label in zip(articles_list,labels):
        event_entries.append((label,event_result['date'].strftime("%Y-%m-%d"),event_result['category'],event_result['location'],article_path))
     
      event_entries_tup = tuple(event_entries) 
      pprint(event_entries_tup)
      con = sqlite3.connect('/home/dvc2106/newsblaster_project/binaryNLP/web/events_id.db')
     
      with con:
    
        cur = con.cursor()    
        #cur.execute("CREATE TABLE Clustering_Events(EventId INT, Date TEXT, Category TEXT,Location TEXT, Path TEXT)")
        cur.executemany("INSERT INTO Clustering_Events (EventId,Date,Category,Location,Path) VALUES(?, ?, ?, ?, ?)", event_entries_tup)

    return labels, new_X

  def _aggr_by_labels(self,labels,tfidf_vectors,articles):

    articles_updated = defaultdict(list)
    articles_by_label = defaultdict(list)

    #tf_idf by date , category and label
    for label, tfidf_vector, article in zip(labels,tfidf_vectors,articles):
      articles_by_label[label].append(tfidf_vector.toarray())
      articles_updated[label].append(article)

    # Drop clusters with  less than 3 documents
    for label in articles_by_label.keys():
      if (len(articles_by_label[label]) < 3):
        del articles_by_label[label]
        del articles_updated[label]

    #Identify and remove duplicate articles
    #for label in articles_by_label.keys():
    #  article_vectors = articles_by_label[label]
    #  num_vec = len(article_vectors)
    #  for ix in range(num_vec):
    #    iy = ix + 1
    #    if(iy <= num_vec - 1 ):
    #      for iz in range(iy,num_vec -1):
    #        print ix
    #        print iz
    #        cosine_scores = cosine_similarity(article_vectors[ix].flatten(),article_vectors[iz])
    #        pprint(cosine_scores)


    return articles_by_label, articles_updated

  def gen_tf_idf(self,arg_list):
 
    # - Unpack variables
    job_queue = arg_list[0]
    debug = arg_list[1] 

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
      event_result['location'] = location

      articles_list = []
      for article in articles:
        articles_list.append(article['file'])

      # learn vocab
      self.vec.fit(articles_list)

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
      event_result['location'] = location

      labels, tfidf_vectors = self._identify_events(articles,event_result,debug)
      num_events = len(set(labels))
    
      #print "*****"
      #print len(labels)
      #print len(tfidf_vectors)
      #print "*****"

      # Sanity check
      if(len(labels) != len(tfidf_vectors)):
        raise Exception("Number of labels and tf-idf vectors do not match")
     
 
      event_result['num_events'] = num_events
      event_result['labels'] = labels

      articles_by_labels = defaultdict(list)
      #tf_idf by date , category and label
      for label, tfidf_vector in zip(labels, tfidf_vectors):
        #print tfidf_vector
        #print tfidf_vector.flatten()
        #print "******"
        articles_by_labels[label].append(tfidf_vector.flatten())  
        
 
      # Average  cluster article tf_idfs
      events_tfidf = defaultdict(dict)
      for cluster_label in articles_by_labels:
        
        #if (len(articles_by_labels[cluster_label]) == 4 and category == 'sci_tech'):
        #  print category
        #  print cluster_label  
        #  pprint(articles_by_labels[cluster_label])
        #  print words
        #sys.exit(0)

        #print articles_by_labels[cluster_label]
        tfidf_avg = np.mean(articles_by_labels[cluster_label],axis=1)
        events_tfidf[cluster_label] = tfidf_avg
          

      event_result['events'] = events_tfidf 
      #TODO average intra-cluster similarity


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
    tf_queue = mgm.Queue()
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
          tf_queue.put((articles_by_location[location],category,article_date,location))


    # Generate TF-IDFs
    print 'Starting TF-IDFn...'
    pool = []
    for i in xrange(n_procs):
      p = multiprocessing.Process(
      target=self.gen_tf_idf, args=((tf_queue,result_queue,False),),)
      p.start()
      pool.append(p)

    for p in pool:
      p.join()
  


    # Start event identification 
    print 'Starting Event Identification...'

    # - Pool
    pool = []
    for i in xrange(n_procs):
      p = multiprocessing.Process(
      #target=self.count_events, args=((job_queue,result_queue,True),),)
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

