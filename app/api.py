from flask import Flask, render_template, request, jsonify
#import requests
from flask_mysqldb import MySQL
import os, json
import datetime

app = Flask(__name__)
app.run(debug=True)

@app.route('/')
def index():
    return render_template('./upload.html')

@app.route('/identification', methods=['POST', 'GET'])
def success():
	if request.method == 'POST':
		t=str(datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
		path="uploads/"
		f = request.files['file']
		#if not os.path.exists(path):
		#	os.mkdir(path)
		
		#call validation function
		speaker_id=0
		username=""
		total_time=""

		f.save(os.path.join(path, t+".wav"))
		res = jsonify(
        value="file saved in "+os.path.join(path, t+".wav"),
        status=200,
        mimetype='application/json'
    	)
		return res
	else:
		return render_template('./upload.html')

	
'''
@app.route('/forward', methods=['POST'])
def forward():
    print("Posted file: {}".format(request.files['audio']))
    file = request.files['audio']
    files = {'file': file.read()}
    r = requests.post("http://127.0.0.1:8000/handle/", files=files) #end point for the next server

    if r.ok:
        return "File uploaded!"
    else:
        return "Error uploading file!"

'''

