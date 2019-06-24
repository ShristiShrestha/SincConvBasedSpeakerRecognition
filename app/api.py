from flask import Flask, render_template, request, jsonify
#import requests
import os, json
import datetime
from test import test
import pymysql.cursors
from database import *

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./upload.html')
#200 success ,401 no credential 400 bad req 403 noo permission to see the res

@app.route('/identification', methods=['POST', 'GET'])
def success():
	if request.method == 'POST':
		t=str(datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
		path="uploads/"
		f = request.files['file']
		print(t)
		if not os.path.exists(path):
			os.mkdir(path)
		
		#call validation function
		speaker_id=0
		username=""
		total_time=""

		f.save(os.path.join(path, t+".flac"))
		result = test(os.path.join(path, t+".flac"))
	
		print(result)
		res = jsonify(user_id=int(result),
        	status=200,
        	mimetype='application/json'
    		)
		x = retrieve_data(1)
		return res
	else:
		return render_template('./upload.html')

@app.route('/insert', methods=['POST', 'GET'])
def insert_to_db():
	if request.method == 'POST':
		audio_path="audio_upload/"
		image_path = "image_upload/"
		audio = request.files['audio']
	
		image = request.files['image']
		user_name = request.form.get('uname')
		user_id = request.form.get('user_id')
		audio.save(os.path.join(audio_path, str(user_id)+".wav"))
		image.save(os.path.join(image_path, str(user_id)+".jpg"))
		res = connect_to_db(user_name, audio_path+str(user_id)+".wav",image_path+str(user_id)+".jpg")
		res = jsonify(message = res,
        	status=200,
        	mimetype='application/json'
    		)
	else:
		res = jsonify(message = "Use post instead of get",
        	status=200,
        	mimetype='application/json'
    		)
	return res




	

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

if __name__=='__main__':
	app.run(debug=True,port = 5000)
