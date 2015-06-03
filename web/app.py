# import the Flask class from the flask module
from flask import Flask, render_template
import sqlite3
from pprint import pprint
from flask import jsonify
from flask import request


#Hacking stuff being done
#We should be using Ids in db and  the works but since this is one off code for cluster evaluation

# Static directory path
data_path = "/Users/skillachie/Documents/BinaryNLP/new/"


def is_new_event(rows):

	new_rows = []

	for row in rows:
		curr_index = rows.index(row)
		if(curr_index-1 != 0):
			pre_index =  curr_index - 1

			pre_item = rows[pre_index] 
			eventid = pre_item[0]
			category = pre_item[1]
			location = pre_item[3]
			date = pre_item[4]

			if(eventid == row[0] and category == row[1] and location == row[3] and date == row[4]):
				new_rows.append((row[0],row[1],row[2],row[3],row[4],'old'))
			else:
				new_rows.append((row[0],row[1],row[2],row[3],row[4],'new'))
		else:
			new_rows.append((row[0],row[1],row[2],row[3],row[4],'new'))
		
	return new_rows

def connect_db():
	print"making db call"
	con = sqlite3.connect('events.db')
	cur = con.cursor()    
	cur.execute("SELECT EventId, Category,Path,Location,Date FROM Clustering_Events ORDER BY Date ASC, Category ASC, EventId ASC")
	rows = cur.fetchall()
	new_rows = is_new_event(rows)
	#pprint(new_rows)
	return new_rows

def insert_eval(eventid,category,location,date,selection):
	con = sqlite3.connect('events.db')
	with con:
		cur = con.cursor() 
		cur.execute("INSERT INTO Clustering_Events_Evaluation VALUES(?,?,?,?,?)", (eventid, category,location,date,selection))
		print category
		print selection



# create the application object
app = Flask(__name__)


@app.route("/clusterEval")
def updateClusterEval():
	 eventid = request.args.get('event_id')
	 category = request.args.get('category')
	 location = request.args.get('location')
	 date = request.args.get('date')
	 selection = request.args.get('selection')
	 insert_eval(eventid,category,location,date,selection)
	 return jsonify(result="successfully saved evaluation")



@app.route("/file")
def readFile():
	 file_name = request.args.get('file_name')
	 full_path = data_path + file_name
	 fh = open(full_path, "rU")
	 lines = fh.readlines()
	 fh.close()
	 return jsonify(file_name=file_name,
	 				text = lines)

# use decorators to link the function to a url
@app.route('/')
def home():
    return "Hello, World!"  # return a string

@app.route('/welcome')
def welcome():
	items = connect_db()
	return render_template('welcome.html',items=items)  # render a template


# start the server with the 'run()' method
if __name__ == '__main__':
	app.run(debug=True)
