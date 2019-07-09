import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import base64
from flask import send_file
import requests
from PIL import Image
from io import BytesIO
import shutil

class InfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(InfoDialog, self).__init__(parent)
        
        self.setWindowTitle("Recognition")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.nameLabel = QtWidgets.QLabel("")
        self.verticalLayout.addWidget(self.nameLabel)

        self.photoLabel = QLabel(self)
        self.verticalLayout.addWidget(self.photoLabel)
        

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.verticalLayout.addWidget(self.buttonBox)

    def displayInfo(self, data):

        if(data['uname'] == "error"):
            self.nameLabel.setText("Model couldn't recognize you\n\n")
            return

        self.nameLabel.setText(
            "Model predicted you as:\n\n UserName : " + str(data['uname'])
        )

        #print("\n \n "+data['image']+" \n \n")
        #image = base64.decodestring(data['image'])

        user_id = data['uid']

        response = requests.get('http://127.0.0.1:5000/get_image?user_id='+str(user_id), stream=True)


        print(response)
        


        with open('temp_image.jpeg', 'wb') as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
        del response


        #qimg = Image.open(BytesIO(response.content))
        pixmap = QtGui.QPixmap('temp_image.jpeg')

        #qimg = QtGui.QImage.fromData(image_data)
        #pixmap = QtGui.QPixmap.fromImage(qimg)

        #pixmap = QtGui.QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.photoLabel.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        



    def RetrieveData(self, id, acc):
    	data = retrieve_data(id)
    	if data!=None:
	    	nameLabel = QtWidgets.QLabel(data.get("User_Name"))
	    	self.verticalLayout.addWidget(nameLabel)

	    	photoLabel = QLabel(self)
	    	pixmap = QtGui.QPixmap(data.get("Photo_Path"))
	    	pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
	    	photoLabel.setPixmap(pixmap)
	    	self.resize(pixmap.width(), pixmap.height())
	    	self.verticalLayout.addWidget(photoLabel)
    	else:
    		nameLabel = QtWidgets.QLabel("Oops! You are not recognized")
	    	self.verticalLayout.addWidget(nameLabel)
	    	








