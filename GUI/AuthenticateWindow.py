import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

from database import *




class AuthenticateWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(AuthenticateWindow, self).__init__(parent)

        self.setWindowTitle("Authenticate")

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        #self.textBrowser = QtWidgets.QTextBrowser(self)
        #self.textBrowser.append("This is a QTextBrowser!")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        #self.verticalLayout.addWidget(self.textBrowser)

        #self.RetrieveData(5)

        self.Input()

        self.verticalLayout.addLayout(self.inputLayout)
        self.verticalLayout.addWidget(self.buttonBox)

        


    def Input(self):
        self.inputLayout = QtWidgets.QHBoxLayout(self)
        






    def RetrieveData(self, id):
    	data = retrieve_data(id)
    	if data!=None:
    		print(data)

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
	    	


