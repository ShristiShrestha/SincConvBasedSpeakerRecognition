import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

from SendEmail import *
from FinalWindow import *


class AuthenticateWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(AuthenticateWindow, self).__init__(parent)

        self.setWindowTitle("Authenticate")

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

        


    def Input(self):
            self.inputLayout = QtWidgets.QHBoxLayout(self)

    def getText(self):
        text, okPressed = QInputDialog.getText(self, "Get text", "Enter OTP:", QLineEdit.Normal, "")
        if okPressed and text != '':
            return text;
        

    def displayInfo(self, data, authorizedUser, password):

        if(data['uname'] == "error"):
            self.nameLabel.setText("Authentication Failed.\n\nTry again...")
            return

        user_id = data['uid']

        if(user_id!=authorizedUser):
            self.nameLabel.setText("You are not authorized.\n\nTry again...")
            self.nameLabel.setStyleSheet('color: red')
        else:
            newOtp = generateOTP()
            sendEmail(newOtp, password)

            enteredOtp = self.getText()

            while (enteredOtp != newOtp):
                enteredOtp = self.getText()

            if newOtp == enteredOtp:
                self.nameLabel.setStyleSheet('color: black')
                #self.SplashScreen(str(data['uname']))

                self.finalBrowser = FinalDialog(self)
                self.finalBrowser.displayInfo(data['uname'])
                self.finalBrowser.show()
                self.finalBrowser.exec_()

            else:
                self.nameLabel.setText("OTP incorrect.\n\nTry again...")
                self.nameLabel.setStyleSheet('color: red')
                return

        # self.nameLabel.setText(
        #     "Model predicted you as:\n\n UserName : " + str(data['uname'])
        # )


        """
        response = requests.get('http://127.0.0.1:5000/get_image?user_id='+str(user_id), stream=True)


        with open('temp_image.jpeg', 'wb') as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
        del response

        pixmap = QtGui.QPixmap('temp_image.jpeg')
        """

        #qimg = Image.open(BytesIO(response.content))

        #qimg = QtGui.QImage.fromData(image_data)
        #pixmap = QtGui.QPixmap.fromImage(qimg)

        #pixmap = QtGui.QPixmap.fromImage(qimg)
        """
        pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.photoLabel.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        """

    
    #not used
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
	    	


    def SplashScreen(self, username):
        import sys, time

        app = QtWidgets.QApplication(sys.argv)


        #Welcome screen
        splash_pix = QtGui.QPixmap('img/splash.png')

        splash_pix = splash_pix.scaled(512, 512, QtCore.Qt.KeepAspectRatio)

        splash = QtWidgets.QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.setEnabled(False)

        progressBar = QtWidgets.QProgressBar(splash)
        progressBar.setMaximum(10)
        progressBar.setGeometry(0, splash_pix.height() - 50, splash_pix.width(), 20)

        splash.show()
        splashMessage = "<h1>Welcome </h1>" + username + "." 
        splash.showMessage(splashMessage, Qt.AlignTop | Qt.AlignCenter, Qt.black)


        for i in range(1, 11):
            progressBar.setValue(i)
            t = time.time()
            while time.time() < t + 0.1:
                app.processEvents()

        time.sleep(2)


        splash.finish(None)

        app.exec_()
