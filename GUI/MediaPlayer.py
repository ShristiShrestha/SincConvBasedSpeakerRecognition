from PyQt5 import QtMultimedia
import sys
import os

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
 
def playAudio(filename):
	url= QtCore.QUrl.fromLocalFile(filename)
	content= QtMultimedia.QMediaContent(url)
	player = QtMultimedia.QMediaPlayer()
	player.setMedia(content)
	player.play()

	return player



def getButton(btnName, audioPath):
	btn = QtWidgets.QPushButton(btnName)
	btn.setIcon(QtGui.QIcon("img/playBtn.png"))
	btn.setToolTip('Play audio.')
	playLabel = QtWidgets.QLabel()
	@pyqtSlot()
	def on_play_button_click():
		cwd = os.getcwd()
		print(audioPath)
		playAudio(audioPath)

	btn.clicked.connect(on_play_button_click)
	return btn