import sys
import os
from sklearn.externals import joblib
from pprint import pprint

class TxtClassificationModel(object):


	#TODO add sub category model for us and world_locatoin
	def __init__(self):
		self.class_loc = os.path.dirname(__file__)

		self.class_loc = os.path.abspath(self.class_loc) + '/'
		self.load_tfidf()
		self.load_clf()
		self.load_le()

	def load_tfidf(self):
		self.vec = joblib.load(self.class_loc + 'tfidf_vectorizer.pkl')

	def load_clf(self):
		self.clf = joblib.load(self.class_loc + 'all_categories.pkl')

	def load_le(self):
		self.le = joblib.load(self.class_loc+ 'label_encoder.pkl')

	def _predict_labels_only(self,files):
		X = self.vec.transform(files)
		y_pred = self.clf.predict(X)
		return(self.le.inverse_transform(y_pred))


	def predict_labels(self,files,date):
		predictions = []
		y_preds = self._predict_labels_only(files)
		for file_path,y_pred in zip(files,y_preds):
			file_meta = file_path.split('/')
			predictions.append({'date':date,'category':y_pred,'file':file_path})

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
		'/2011-01-03-04-19-00/www.foxnews.com.9235.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.6190.txt',
		'/2011-01-03-04-19-00/www.foxnews.com.17000.txt']

	files_list = []
	for file_name in files:
		files_list.append(path + file_name)

	txt_classifier = TxtClassificationModel()
	pprint(txt_classifier.predict_labels(files_list,directory))
