import sys
import os
from sklearn.externals import joblib
from pprint import pprint

class TxtClassificationModel(object):


	def __init__(self):
		self.class_loc = os.path.dirname(__file__)

		self.class_loc = os.path.abspath(self.class_loc) + '/'
		self.load_tfidf()
		self.load_clf()
		self.load_le()

	def load_tfidf(self):
		self.cat_vec = joblib.load(self.class_loc + 'tfidf_vectorizer.pkl')
		self.loc_vec = joblib.load(self.class_loc + 'location_tfidf_vectorizer.pkl')

	def load_clf(self):
		self.cat_clf = joblib.load(self.class_loc + 'all_categories.pkl')
		self.loc_clf = joblib.load(self.class_loc + 'location_clf.pkl')

	def load_le(self):
		self.cat_le = joblib.load(self.class_loc+ 'label_encoder.pkl')
		self.loc_le = joblib.load(self.class_loc+ 'location_label_encoder.pkl')

	def _predict_labels_only(self,files):
		X = self.cat_vec.transform(files)
		y_pred = self.cat_clf.predict(X)
		return(self.cat_le.inverse_transform(y_pred))

	def _predict_loc(self,files):
		X = self.loc_vec.transform(files)
		y_pred = self.loc_clf.predict(X)
		return(self.loc_le.inverse_transform(y_pred))
		
	def predict_labels(self,files,date):
		predictions = []
		y_cat_preds = self._predict_labels_only(files)
		y_loc_preds = self._predict_loc(files)

		for file_path,y_cat_pred,y_loc_pred in zip(files,y_cat_preds,y_loc_preds):
			predictions.append({'date':date,'location':y_loc_pred,'category':y_cat_pred,'file':file_path})

		return(predictions)

	def predict_labels_test(self,labels,files):
		if len(labels) != len(files):
			raise Exception('Number of categories not equal to the number of files')
		# TODO used to test how good the model is with
		# labeled unseen data

#TODO load the other & world model
if __name__ == '__main__':

	class_loc = os.path.dirname(__file__)
	path = os.path.abspath(class_loc) + '/../data'
	pprint(path)
	directory = '2011-01-03-04-19-00'
	files =  [ '/2011-01-03-04-19-00/www.foxnews.com.9225.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.492.txt',
		'/2011-01-03-04-19-00/www.cbc.ca.9364.txt',
		'/2011-01-03-04-19-00/www.haaretz.com.1096.txt',
		'/2011-01-03-04-19-00/www.baltimoresun.com.9867.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.5858.txt',
		'/2011-01-03-04-19-00/www.washingtonpost.com.10010.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.9235.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.6190.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.17000.txt']

	files_list = []
	for file_name in files:
		files_list.append(path + file_name)

	txt_classifier = TxtClassificationModel()
	pprint(txt_classifier.predict_labels(files_list,directory))
