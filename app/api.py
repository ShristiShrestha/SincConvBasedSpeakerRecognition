from flask import Flask, render_template, request, jsonify
#import requests
import os, json
import datetime, time
from test import test,validation_test,train
import pymysql.cursors
from database import *
import base64
import random
from flask import send_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./upload.html')
#200 success ,401 no credential 400 bad req 403 noo permission to see the res


#endpoint for identification send data in multipart form
@app.route('/identification', methods=['POST', 'GET'])
def success():
	if request.method == 'POST':
		t=str(datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
		path="uploads/"
		f = request.files['file']

		if not os.path.exists(path):
			os.mkdir(path)

		f.save(os.path.join(path, t+".wav"))
		result,accuracy = test(os.path.join(path, t+".wav"))

		if int(result)>=251 and int(result)<=264:
			x = retrieve_data(int(result))
			with open(x['Photo_Path'],"rb") as imageFile:
				str_image = base64.encodestring(imageFile.read())
			image = {'uname':x['User_Name'],'uid':x['User_ID'],'acc':str(accuracy),'status':200,'mimetype':'application/json'}
			res = jsonify(image)
			return res
		elif int(result)>0 and int(result)<=250:
			return jsonify({'uname':"error"})
		else:
			return jsonify({'uname': "internal_error",'status':500})


@app.route('/demo', methods=['POST', 'GET'])
def demos():
	if request.method == 'POST':
		image = {'uname':" ",'uid': random.randint(1, 100),'acc':random.uniform(0.05, 0.2),'status':200,'mimetype':'application/json'}
		res = jsonify(image)
		time.sleep(10)
		print (res)
		return res



#api endpoints for validation send uname and audio in a form
@app.route('/validation', methods=['POST', 'GET'])
def validates():
	if request.method == 'POST':
		t=str(datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
		path="uploads/"
		f = request.files['file']

		if not os.path.exists(path):
			os.mkdir(path)

		f.save(os.path.join(path, t+".wav"))
		result,accuracy = validation_test(os.path.join(path, t+".wav"))

		if result == 0:
			#when threshhold is not reached
			return jsonify({'uname':"error"})
		elif result>=251 and result<=264:
			x = retrieve_data(int(result))
			with open(x['Photo_Path'],"rb") as imageFile:
				str_image = base64.encodestring(imageFile.read())
			image = {'uname':x['User_Name'],'uid':x['User_ID'],'acc':str(accuracy),'status':200,'mimetype':'application/json'}
			res = jsonify(image)
			return res
		elif int(result)>0 and int(result)<=250:
			return jsonify({'uname':"error"})
		else:
			return jsonify({'uname': "internal_error",'status':500})


#api endpoints for training 
@app.route('/training', methods=['GET'])
def trains():
	if request.method == 'GET':
		train()
		return jsonify({'status': 200})

		
#api endpoint to insert new user in db requires audio,image and uname
@app.route('/insert', methods=['POST', 'GET'])
def insert_to_db():
	if request.method == 'POST':
		audio_path="audio_upload/"
		image_path = "image_upload/"
		audio = request.files['audio']
		image = request.files['image']
		user_name = request.form.get('uname')
		user_id = request.form.get('user_id')
		if not os.path.exists(audio_path+str(user_id)):
			os.makedirs(audio_path+str(user_id))
		audio.save(os.path.join(audio_path+str(user_id),"audio.wav"))
		image.save(os.path.join(image_path, str(user_id)+".jpg"))
		res = connect_to_db(user_name, audio_path+str(user_id)+"audio.wav",image_path+str(user_id)+".jpg")
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

#api end point to exract image to display 
@app.route('/get_image', methods=['GET'])
def get_image():
	if request.method == 'GET':
		id = request.args.get('user_id')

		x = retrieve_data(int(id))
		image_path = x['Photo_Path']

		return send_file(image_path , mimetype='image/jpg')




if __name__=='__main__':
	app.run(debug=True,port = 5000)


