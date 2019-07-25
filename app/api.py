from flask import Flask, render_template, request, jsonify
#import requests
import os, json
import datetime
from test import test
import pymysql.cursors
from database import *
import base64
from flask import send_file

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

		f.save(os.path.join(path, t+".wav"))
		result = test(os.path.join(path, t+".wav"))
		print(result)	
		if int(result)>=251 and int(result)<=261:
			x = retrieve_data(int(result))
			with open(x['Photo_Path'],"rb") as imageFile:
				str_image = base64.encodestring(imageFile.read())
			image = {'uname':x['User_Name'],'uid':x['User_ID'],'status':200,'mimetype':'application/json'}
			res = jsonify(image)
			return res
		else:
			return jsonify({'uname': "error"})
			
	#else:
	#	return render_template('./upload.html')

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



@app.route('/get_image', methods=['GET'])
def get_image():
	if request.method == 'GET':
		id = request.args.get('user_id')

		x = retrieve_data(int(id))
		image_path = x['Photo_Path']

		return send_file(image_path , mimetype='image/jpg')



if __name__=='__main__':
	app.run(debug=True,port = 5000)


