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

class FinalDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(FinalDialog, self).__init__(parent)
        
        self.setWindowTitle("Welcome")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.nameLabel = QtWidgets.QLabel("")
        self.verticalLayout.addWidget(self.nameLabel)

        self.infoLabel = QtWidgets.QLabel("")
        self.verticalLayout.addWidget(self.infoLabel)

        self.photoLabel = QLabel(self)
        self.verticalLayout.addWidget(self.photoLabel)
        

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.verticalLayout.addWidget(self.buttonBox)

    def displayInfo(self, username):

        self.nameLabel.setText(
            "Welcome " + str(username)
        )

        self.infoLabel.setText(
            "You are successfully authenticated"
            )

        splash_pix = QtGui.QPixmap('img/splash.png')

        splash_pix = splash_pix.scaled(512, 512, QtCore.Qt.KeepAspectRatio)

        self.photoLabel.setPixmap(splash_pix)

        """
        pixmap = QtGui.QPixmap('temp_image.jpeg')

        pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.photoLabel.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        """



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
	    	








