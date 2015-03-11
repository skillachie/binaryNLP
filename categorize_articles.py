import os
from pprint import pprint
import re
import tarfile
import sys
import datetime
import time
import subprocess
from model.model import TxtClassificationModel
from multiprocessing import Pool,Manager
import  multiprocessing
from multiprocessing.pool import ThreadPool
from shutil import copyfile

class CategorizeArticles(object):
	"""
	Script used to read articles from the filesystem
	and categorize them using the generated model

	hacked together right now

	"""

	def __init__(self):

		#TODO accept path, startdate, enddate as arguments
		self.start_date = datetime.datetime(2009, 1, 1)
		self.end_date = datetime.datetime(2014, 12, 31)
		self.date_match = re.compile('\d+-\d+-\d+-\d+-\d+-\d+')


		#path = "/home/dvc2106/newsblaster_project/nb_migration/stream/"
		path = "/home/dvc2106/newsblaster_project/binaryNLP/data/"
		extraction_path = "/home/dvc2106/newsblaster_project/nb_migration/stream/"

		stream_cat_results_file = extraction_path + "stream_cat_results.txt"
		article_list_results_file = extraction_path + "article_list.txt"

		stream_cat_results = []
		#temp
		article_list = []

		directories = os.listdir(path)


	#def run(self,path,categorize_articles,save_results):
	def run(self,path):

		#TODO argument
		start_date = datetime.datetime(2009, 1, 1)
		end_date = datetime.datetime(2014, 12, 31)
		date_match = re.compile('\d+-\d+-\d+-\d+-\d+-\d+')


		#TODO use queue pattern to speed up the process
		#/home/dvc2106/newsblaster_project/corenlp/corenlp/client.py
		# TODO later figure out a way to share the model

		#txt_classifier = TxtClassificationModel()
		cpu_count = multiprocessing.cpu_count()

		# - Hardcode to 2 until the model is passed as shared object
		pool = Pool(2)

		directories = os.listdir(path)

		mgm = Manager()
		prediction_results = mgm.list()

		for directory in directories:
			pool.apply_async(categorize_articles, args=((directory,date_match,start_date,end_date,path,prediction_results),))
			#pool.apply_async(categorize_articles, args=((directory,date_match,start_date,end_date,path,prediction_results),),callback=save_results)
		pool.close()
		pool.join()

		return prediction_results

		#print "here .."
		
		# - Check results
	#	for result in prediction_results:
	#		pprint(result)

		#pprint(prediction_results)
		#pool.macategorize_articles,[(directories,date_match,start_date,end_date)])
		#TODO - parralize code here 

#prediction_results = []

def save_results(result):
	prediction_results.extend(result)

def get_results():
	return prediction_results

def categorize_articles(arg_list):

	# - Unpack variables
	directory = arg_list[0]
	date_match = arg_list[1]
	start_date = arg_list[2]
	end_date = arg_list[3]
	path = arg_list[4]
	prediction_results = arg_list[5]
	#txt_classifier = TxtClassificationModel()
	
	if date_match.match(directory):
		date = datetime.datetime.strptime(directory, '%Y-%m-%d-%H-%M-%S')
		if date >= start_date and date <= end_date:
			pprint(directory)

			# - Path to location of extracted files
			articles_location = path + '/' +  directory 	
			files = []
			if os.path.exists(articles_location):
				#TODO filter for only .txt here
				files = os.listdir(articles_location)
			else:
				raise Exception('File location not valid.',articles_location)
			
			# - Fully qualified file name
			articles_list = []
			for file_name in files:
				articles_list.append(articles_location + '/' + file_name)
			files = None


			# - Initialize text classification model
			#TODO make this a shared object. If passed to each process it is pickled and copied
			# which defeats the purpose
			txt_classifier = TxtClassificationModel()
			
			# - Get predictions
			preds = txt_classifier.predict_labels(articles_list)
			prediction_results.extend(preds)
			return prediction_results
		#	prediction_results.append(preds)

# - Get save list of predictions to be returned once done for all
#sys.exit(1)

#if not os.path.exists(cat_results_path):
	# Do categorization using perl script
	#subprocess.call(["/home/dvc2106/categorization_migration/src/BINS/do_categorizing.pl", file_list_path, cat_results_path])

# Add date of categorization to file
#categorization_results = [line.strip() for line in open(cat_results_path)]	
#for cat_result in categorization_results:
#	stream_cat_results.append(directory + " " + cat_result)
		

def move_files_newsblaster(path):
	'''
	One off script
	'''
	directories = os.listdir(directory_location)
	pool = Pool(20)

	directories = os.listdir(path)

	for directory in directories:
		pool.apply_async(move_file_p, args=((directory,path),))
		#pool.apply_async(categorize_articles, args=((directory,date_match,start_date,end_date,path,prediction_results),),callback=save_results)
	pool.close()
	pool.join()

def move_file_p(args):
		directory = args[0]
		path = args[1]
		print directory
		files_location_2 = path + '/' + directory +   "/proj/nlp/users/blaster/newsblaster/data/ArticleExtractor/clean/"
		files_location_3 = path + '/' + directory +  "/proj/nlpdisk3/nlpusers/blaster/newsblaster/data/ArticleExtractor/clean/"

		files = []
		files_location = ''
		if os.path.exists(files_location_2):
			files = os.listdir(files_location_2)
			files_location = files_location_2
		elif os.path.exists(files_location_3):
			files = os.listdir(files_location_3)
			files_location = files_location_3

		print files_location
		mv_path = path + '/' + directory + '/'
		for file_name in files:
			src = files_location +'/' +file_name
			dst = mv_path + file_name
			os.symlink(src, dst)
			#copyfile(src, dst)

		# Delete old results		
		cat_path = mv_path + 'cat_results_path.txt'
		file_path = mv_path + 'file_list.txt'
		if os.path.exists(cat_path):
			os.remove(cat_path)
		if os.path.exists(file_path):
			os.remove(file_path)
	
if __name__ == '__main__':
	categorizer = CategorizeArticles()
	#manager = CatManager()
	results = categorizer.run('/home/dvc2106/newsblaster_project/binaryNLP/data')
	#results = categorizer.run('/home/dvc2106/newsblaster_project/binarynlp/data',categorize_articles,save_results)
	pprint(results)

#	directory_location = '/home/dvc2106/newsblaster_project/nb_migration/stream'
#	move_files_newsblaster(directory_location)

