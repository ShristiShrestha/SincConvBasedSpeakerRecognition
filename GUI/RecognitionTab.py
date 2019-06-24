import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

class InfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(InfoDialog, self).__init__(parent)
        
        self.setWindowTitle("Recognition")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.nameLabel = QtWidgets.QLabel("")
        self.verticalLayout.addWidget(self.nameLabel)

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.verticalLayout.addWidget(self.buttonBox)

    def displayInfo(self, id):
        self.nameLabel.setText(
            "Model predicted you as:\n\n UserName :" + str(id)
        )

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
	    	








