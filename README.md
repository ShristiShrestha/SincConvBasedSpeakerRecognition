# SincConvBasedSpeakerRecognition

## Run command: 
### > pip install flask
### > py -m venv <venv_name>
### > cd <venv_name>/scripts
### > activate
### >cd ../..
### > set FLASK_APP = api  #api.py run 
### > set FLASK_ENVIRONMENT= development  
### > flask run # run server at 127.0.0.1:5000

## Endpoints
### 127.0.0.1:5000/  => file upload interface, submission and saving the audio file in serverside for validation
 
### using postman (google chrome extension)
### POST: 127.0.0.1:5000/identification in the url
### select form-data (parm), key= file and choose file and hit send 
### return json ( after integration, it will return id, username, % acc, response code)
