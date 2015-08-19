import pandas as pd
from pandas.stats.moments import rolling_mean
from pandas.stats.moments import rolling_std
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint
import numpy as np
import sys
from collections import defaultdict
import datetime
import  time
from pandas.tseries.offsets import BDay
from sklearn.metrics.pairwise import cosine_similarity
import pickle


#TODO remove duplicate code

class BinaryBase(object):

  def _create_series(self,dictionary_series):
    df =  pd.DataFrame.from_dict(dictionary_series,orient='index')
    return df

  def _create_binary_series_nstd(self,series,period):
    mean = rolling_mean(series,period)
    std = rolling_std(series,period) 

    normalized_std = (series - mean)/std
    df = normalized_std
    #pprint(df[(df < 2) & (df >1)])
    significant_df = (df[(df >1)])
    
    # Set all NANs to zero
    tmp_df = significant_df.fillna(0)
    #pprint(tmp_df) 

    # Set all values => 1 to 1 
    tmp_df[(tmp_df >= 1)] = 1

    #Update labels to period
    new_cols = []
    col_names = df.columns.values.tolist()
    for col in col_names:
      new_cols.append(col + "_" + "std" + "_" + str(period))
    
    tmp_df.columns = new_cols
    return tmp_df

  def _create_binary_series_quantile(self,series):
    
    pc = series.pct_change()

    # -Replace any possible inf as a result of no change
    pc = pc.replace([np.inf, -np.inf], np.nan)

    # -Fill forward pct indicating no change was observed
    #pc = pc.fillna(method='ffill')
    #pc = pc.fillna(0)
    #print pc
    dc = pc.apply(pd.qcut,reduce=False,args=(10,[1,2,3,4,5,6,7,8,9,10],False,3,))

    # Fill missing value with lowest quantile
    dc_fl = dc.fillna(1)
    
    # Get bins from all categories
    cat_enc = OneHotEncoder()
    X = cat_enc.fit_transform(dc_fl.as_matrix().astype(np.int32)) 

    # Create dataframe 
    qd = pd.DataFrame(X.todense())

    # Create labels based on values returned 
    col_labels = []
    cat_cols = dc_fl.columns.values.tolist()
    for col_name in cat_cols:
      quantiles = dc_fl[col_name].values
      label_list = []
      for quantile in quantiles:
        label = (col_name ,"_quartile_" , int(quantile))
        label_list.append(label)

      # Sort and Create labels
      labels_sorted = []
      for tup in sorted(set(label_list)):
        labels_sorted.append(tup[0] + tup[1] + str(tup[2]))
      col_labels = np.concatenate([col_labels,labels_sorted],axis=0)      

    # Set category quartile column labels
    qd.columns = col_labels

    # Set Index 
    dt_index = pd.DatetimeIndex(pc.index.values.tolist())
    qd_c = qd.set_index(dt_index)

    return qd_c
  
class CategorySeries(BinaryBase):

  def __init__(self,start,end,aggr_freq='daily',add_noise=False):
    self.category_list = []
    self.aggr_freq = aggr_freq
    self.add_noise = add_noise

    # TODO  accept paramter later
    self.start_date = '2009-01-01'
    self.end_date = '2014-12-31'
    # Business date for range
    self.bus_range = pd.bdate_range(start=self.start_date, end=self.end_date)

    if self.aggr_freq == 'hourly':
      print 'Creating hourly series ..'
      self._set_hourly_bus_hours()


  def _set_hourly_bus_hours(self):
    bus_day_hours = []
    for bday in self.bus_range:
      bday_range = pd.date_range(bday,periods=24, freq="1H")
      bus_day_hours.extend(bday_range)

    self.bus_range = pd.DatetimeIndex(bus_day_hours)


  def _aggr_predictions(self,predictions,series_type=None):
    #TODO vary aggregation based on type

    print 'Aggregating results by date...'    

    date_by_category = defaultdict(dict)
    for prediction in predictions:

       
        date = datetime.datetime.strptime(prediction['date'], '%Y-%m-%d-%H-%M-%S')

        if self.aggr_freq == 'hourly':
          hour = datetime.time(date.hour)
          short_date = datetime.date(date.year, date.month, date.day)
          date = datetime.datetime.combine(short_date, hour)
        else:
          date = datetime.date(date.year, date.month, date.day)

        #predicted_category = prediction['category']
        predicted_category = prediction['category'] + '_location_' + prediction['location']
    
        if date in date_by_category:
          date_by_category[date][predicted_category] += 1
        else:
          predicted_categories_count = defaultdict(int)
          predicted_categories_count[predicted_category] += 1
          date_by_category[date] = predicted_categories_count
  

        # Special aggregation to include cluster indices for events
        if series_type == 'events':
          date_by_category[date][predicted_categories_count]['cluster_center_indices'] = prediction['cluster_center_indices']


    return date_by_category


  #TODO seperate into category_timeseries & event_timeseries methods
  def get_category_timeseries(self,predictions):
    dated_categories = self._aggr_predictions(predictions)

    series = self._create_series(dated_categories)
    series = series.fillna(0)

    print series.head()
    print 'head is done' 
    # For debug
    series.to_csv('raw_topic_counts.csv')

    if self.add_noise:
      pd_matrix = series.as_matrix()
      noise = np.random.normal(size=pd_matrix.shape)
      noise_m = pd_matrix + noise
      series = pd.DataFrame(noise_m,series.index,series.columns.values)
      series.to_csv('raw_topic_counts_noise.csv')


  
    #TODO move business filer to seperate method and base class 
    # Filter on only business days
    to_drop = []
    for day_count in xrange(0,len(series.index)):
      if series.index[day_count] not in self.bus_range:
        # Move values forward for this day to the next
        # Cant use standard pandas functions since it will shit entire dataset
      
        cur_index = series.index[day_count]
        
        #Indices to drop
        to_drop.append(cur_index)
      
        # - If last index break and drop  
        nxt = day_count + 1
        if (nxt >= len(series.index)):
          break

        next_index = series.index[day_count + 1]
        cur_day = series.ix[cur_index]  
        next_day = series.ix[next_index]

        cumm_day = cur_day + next_day
      
        # Update next day index with new values
        series.ix[next_index]  = cumm_day
        

    # Drop weekend indices
    series.drop(to_drop,inplace=True)
    print 'passed business day filter'
    print series


    quantile_series = self._create_binary_series_quantile(series)
    return quantile_series


  #Thread
  def _check_if_continuous(self,article_tfidf,dates,events,clust_label,curr_event_id,prev_date):
    

    continuos_count = 0
    stop_count = 0
    continuous_events_list = []
    stop_events_list = []

    for ev_date in dates:
      #print clust_label
      #pprint(events[ev_date].keys())

      # Only check for continuos events within the same category
      if(clust_label not in events[ev_date].keys()):
        continue


      date_tfidfs = events[ev_date][clust_label]
      if not date_tfidfs:
        #print("%s %s %s %s" %(ev_date, clust_label,prev_date,curr_event_id))
        stop_events_list.append((str(prev_date) + '_' + clust_label + '_' + str(curr_event_id) , ))
        stop_count += 1


      # Iterate each event
      for date_tfidf in date_tfidfs:
        prev_event = date_tfidfs[date_tfidf]


        print article_tfidf.shape
        pprint(article_tfidf)
        print prev_event.shape
        pprint(prev_event)

        # All events might not have the same number of occurrences/documents 
        if (prev_event.shape[0] < article_tfidf.shape[0] ):
            article_tfidf = np.resize(article_tfidf,prev_event.shape)
            #print article_tfidf.shape

        if (prev_event.shape[0] > article_tfidf.shape[0] ):
          #print "resizing..... 2"
          prev_event = np.resize(prev_event,article_tfidf.shape)

        cosine_scores = cosine_similarity(article_tfidf, prev_event)
        #print("%s %s %s %s %s" %(ev_date, clust_label, date_tfidf,prev_date,curr_event_id))
        print cosine_scores

        if(cosine_scores[[0]] >= 0.75):
          continuous_events_list.append((str(prev_date) + '_' + clust_label + '_' + str(curr_event_id) , str(ev_date) + '_' + clust_label +'_' + str(date_tfidf)))
            
          #print "****"
          #print cosine_scores
          #print("%s %s %s %s %s" %(ev_date, clust_label, date_tfidf,prev_date,curr_event_id))
          #print "Continuos event..."
          continuos_count += 1
        else:

          #print cosine_scores
          #print("%s %s %s %s %s" %(ev_date, clust_label, date_tfidf,prev_date,curr_event_id))
          print "Stop event..."
          stop_events_list.append((str(prev_date) + '_' + clust_label + '_' + str(curr_event_id) , str(ev_date) + '_' + clust_label +'_' + str(date_tfidf)))
          stop_count += 1

    return continuos_count,continuous_events_list, stop_count, stop_events_list
        

  def get_continuous_events_timeseries(self,predictions,lookback_period=5,lookback_type='hourly'):


    event_predictions = defaultdict(dict)
    continuous_events_list = []
    stop_events_list = []

    for prediction in predictions:
      label = prediction['category'] + '_' + prediction['location'] + '_events'
      date = prediction['date']
      

      event_predictions[date][label] = prediction['events']


    # Check if each event is continuous or stop per event
    events_duration = defaultdict(dict)

    # - Iterate over sorted dates
    dict_dates = sorted(event_predictions.keys())
    for  date in dict_dates:

      # Check if each event is continuous or stop per event
    #  events_duration = defaultdict(dict)
     # events_duration_count = defaultdict(int)


      # - Get lookback dates
      date_index = dict_dates.index(date)
      lookback_dates = dict_dates[:date_index]
      if(date_index - lookback_period > 0):
        lookback_dates = dict_dates[date_index - lookback_period:date_index]

      # -If lookback period does not exist for start
      if (len(lookback_dates) == 0):
        continue

      # - Get tf-idf vector by labels for that day
      for cluster_label in event_predictions[date]:
        events_duration_count = defaultdict(int)

        events = event_predictions[date][cluster_label]
        #pprint(events) 
        # Check if each event is continuous or stop per event
        #events_duration = defaultdict(int)

        for event in events:

          continuous_count,continuous_events, stop_count, stop_events = self._check_if_continuous(events[event],lookback_dates,event_predictions,cluster_label,event,date)

          if(continuous_count != 0):
            continuous_events_list.extend(continuous_events)


          if(stop_count != 0):
            stop_events_list.extend(stop_events)

          events_duration_count['countinuos'] += continuous_count
          events_duration_count['stop'] += stop_count

        events_duration[date][cluster_label + '_countinuous'] = events_duration_count['countinuos']  
        events_duration[date][cluster_label + '_stop'] = events_duration_count['stop']  

    #pprint(events_duration)   
    #pprint(continuous_events_list)
    #pprint(stop_events_list)
    series = self._create_series(events_duration)
    series = series.fillna(0)
    print series.head()

    if self.add_noise:
      print 'adding noise'
      pd_matrix = series.as_matrix()
      noise = np.random.normal(size=pd_matrix.shape)
      noise_m = pd_matrix + noise
      series = pd.DataFrame(noise_m,series.index,series.columns.values)

    
    to_drop = []
    for day_count in xrange(0,len(series.index)):
      if series.index[day_count] not in self.bus_range:
        # Move values forward for this day to the next
        # Cant use standard pandas functions since it will shit entire dataset
      
        cur_index = series.index[day_count]
        
        #Indices to drop
        to_drop.append(cur_index)
      
        # - If last index break and drop  
        nxt = day_count + 1
        if (nxt >= len(series.index)):
          break

        next_index = series.index[day_count + 1]
        cur_day = series.ix[cur_index]  
        next_day = series.ix[next_index]

        cumm_day = cur_day + next_day
      
        # Update next day index with new values
        series.ix[next_index]  = cumm_day
        

    # Drop weekend indices
    series.drop(to_drop,inplace=True)
    print 'passed business day filter'
    print series.head()

    quantile_series = self._create_binary_series_quantile(series)
    return quantile_series
    #return events_duration
    

  #TODO seperate into category_timeseries & event_timeseries methods
  def get_event_timeseries(self,predictions):


    event_predictions = defaultdict(dict)
    event_indices = []
    for prediction in predictions:
      label = prediction['category'] + '_' + prediction['location'] + '_events'
      date = prediction['date']
      event_predictions[date][label] = prediction['num_events']
      #TODO make compatible for hourly and daily
      #event_indices = prediction['date']
      #event_predictions['date'] = prediction['date']


    series = self._create_series(event_predictions)
    series = series.fillna(0)
    
    print series.head()
    print 'yaa.....'  
    # For debug
    series.to_csv('raw_topic_counts.csv')

    if self.add_noise:
      pd_matrix = series.as_matrix()
      noise = np.random.normal(size=pd_matrix.shape)
      noise_m = pd_matrix + noise
      series = pd.DataFrame(noise_m,series.index,series.columns.values)
      series.to_csv('raw_topic_counts_noise.csv')



    #TODO move business filer to seperate method and base class 
    # Filter on only business days
    to_drop = []
    for day_count in xrange(0,len(series.index)):
      if series.index[day_count] not in self.bus_range:
        # Move values forward for this day to the next
        # Cant use standard pandas functions since it will shit entire dataset
      
        cur_index = series.index[day_count]
        
        #Indices to drop
        to_drop.append(cur_index)
      
        # - If last index break and drop  
        nxt = day_count + 1
        if (nxt >= len(series.index)):
          break

        next_index = series.index[day_count + 1]
        cur_day = series.ix[cur_index]  
        next_day = series.ix[next_index]

        cumm_day = cur_day + next_day
      
        # Update next day index with new values
        series.ix[next_index]  = cumm_day
        

    # Drop weekend indices
    series.drop(to_drop,inplace=True)
    print 'passed business day filter'
    print series


    quantile_series = self._create_binary_series_quantile(series)
    return quantile_series
if __name__ == "__main__":
  category_series = CategorySeries()
  #category_list = category_series.get_category_timeseries(results,'hourly')
