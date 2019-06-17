import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

from database import retrieve_data




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

        #self.textBrowser = QtWidgets.QTextBrowser(self)
        #self.textBrowser.append("This is a QTextBrowser!")

        
        #self.verticalLayout.addWidget(self.textBrowser)

        #self.displayInfo(self.best_class,self.acc)

        self.verticalLayout.addWidget(self.buttonBox)

        

    def displayInfo(self, id, acc):
        self.nameLabel.setText(
            "Model predicted you as:\n\n UserName :" + str(id) + "\nWith\n\nAccuracy :" +str(acc)+"\n"
        )


    def RetrieveData(self, id, acc):
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
	    	








# def Recognition(object):

# 	def __init__(self, parent):

# 		self.hbox_widget = QtWidgets.QWidget()

# 		self.hbox_Recognize = QtWidgets.QHBoxLayout()
# 		SetButtons()

# 		self.layout = QVBoxLayout(self)
# 		self.layout.addLayout(self.hbox_Recognize)


# 		self.hbox_widget.addLayout(hbox_Recognize)




# 	def SetButtons(self):
		
# 		Recognize = QtWidgets.QLabel('Recognize:')
# 		retrieve_button = QtWidgets.QPushButton('Retrieve', self)
# 		retrieve_button.setIcon(QtGui.QIcon("img/retrieveIcon.png"))
# 		retrieve_button.setToolTip('Retrive Data')

# 		@pyqtSlot()
# 		def on_retrieve_button_click(self):
# 			recording_info = QtWidgets.QMessageBox.information(None, "Info!", "Retriv...\n\n")

# 		retrieve_button.clicked.connect(on_retrieve_button_click)
# 		self.hbox_Recognize.addWidget(Recognize)
# 		self.hbox_Recognize.addWidget(retrieve_button)
