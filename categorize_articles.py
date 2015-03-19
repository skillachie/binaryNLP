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

	def __init__(self,
                     match='\d+-\d+-\d+-\d+-\d+-\d+',
                     start=datetime.datetime(2009,1,1),
                     end=datetime.datetime(2014,12,31)):
		#TODO accept path, startdate, enddate as arguments
		self.start_date = start
		self.end_date = end
		self.date_match = re.compile(match)


	def run(self,path,cpu_count = 0):
                if not cpu_count:
                        cpu_count = multiprocessing.cpu_count()
		n_procs = cpu_count -1 if cpu_count > 1 else 1

		directories = os.listdir(path)

		mgm = Manager()
		prediction_results = mgm.list()
		job_queue = mgm.Queue()

    # Add work items to job queue
		for directory in directories:
			job_queue.put(directory)

    # - Pool
		pool = []
		for i in xrange(n_procs):
			p = multiprocessing.Process(
			target=categorize_articles, args=((job_queue, self.date_match,self.start_date,self.end_date,path,prediction_results),),)
			p.start()
			pool.append(p)

		for p in pool:
			p.join()
	
		print "Text Classification complete..."

		return prediction_results

def categorize_articles(arg_list):
# - Unpack variables
	job_queue = arg_list[0]
	date_match = arg_list[1]
	start_date = arg_list[2]
	end_date = arg_list[3]
	path = arg_list[4]
	prediction_results = arg_list[5]

	txt_classifier = TxtClassificationModel()
	while not job_queue.empty():
		directory = job_queue.get(block=False)
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

				# - Get predictions
				preds = txt_classifier.predict_labels(articles_list,directory)
				prediction_results.extend(preds)


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

