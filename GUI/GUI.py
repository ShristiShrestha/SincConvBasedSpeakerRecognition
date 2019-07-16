import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from datetime import datetime
import time
import shutil
import os,pathlib
import soundfile as sf
import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder
from MicRecorder import MicrophoneRecorder
from Interface import MplFigure, CreateFolder
from Recorder import Recorder
from RecognitionTab import InfoDialog
from AuthenticateWindow import AuthenticateWindow
from VarManager import VarManager
from scipy import stats
import collections



def datetime_to_string():
    datetime_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    return datetime_string


class Worker(QRunnable):
    '''
    Worker thread
    '''
    @pyqtSlot()
    def run(self):
        '''
        record
        '''
        micRecorder = Recorder(channels=1)

        datetime_string = datetime_to_string()
        filename = "record_"+datetime_string+".wav"

        with micRecorder.open("records/"+filename, 'wb') as recfile:
            recfile.start_recording()
            while (LiveFFTWidget.stopRecord!=1):
                pass    
            recfile.stop_recording()
            LiveFFTWidget.stopRecord = 0

        VarManager.audioPath = "records/"+filename
        VarManager.audioUploaded = False
        

class RecognitionWorker(QRunnable):
    '''
    Worker thread
    '''
    @pyqtSlot()
    def run(self):
        '''
        record
        '''
        micRecorder = Recorder(channels=1)

        datetime_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = "record_"+datetime_string+".wav"
        with micRecorder.open("records/testing/"+filename, 'wb') as recfile:
            recfile.start_recording()
            time.sleep(10.0)    
            recfile.stop_recording()
            LiveFFTWidget.stopRecord = 0
        
        VarManager.recognitionAudioPath = "records/testing/" + filename


class CountWorker(QtCore.QThread):
    '''
    Worker thread
    '''
    notifyProgress = QtCore.pyqtSignal(int)

    @pyqtSlot()
    def run(self):
        '''
        count
        '''
        count = 0.0
        while count < 10:
            count+=0.1
            self.notifyProgress.emit(count*10)
            time.sleep(0.1)
            VarManager.progressValue = count*10
        


class LiveFFTWidget(QWidget):

    stopRecord = 0
    recording = False
    threadpool = QThreadPool()

    def __init__(self):

        QWidget.__init__(self)

        self.iconName = "img/appIcon.svg"
        CreateFolder("records")

        self.uploadPhotoName = ""
        
        print("Multithreading with maximum %d threads" % LiveFFTWidget.threadpool.maxThreadCount())

        #customize the UI
        self.initUI()
        
        #init class data
        self.initData()       
        
        #connect slots
        self.connectSlots()
        
        #init MPL widget
        self.initMplWidget()


    def generateUserID(self):
        userIdFile = open("userID.uid", "r+")
        userID = int(userIdFile.read())
        userID += 1
        VarManager.userID = userID
        print("User ID: " + str(userID))
        userIdFile.seek(0)
        userIdFile.write(str(userID))
        userIdFile.close()

    def initUI(self):

        self.setTabs()

        hbox_gain = QtWidgets.QHBoxLayout()
        autoGain = QtWidgets.QLabel('Auto gain for frequency spectrum')
        autoGainCheckBox = QtWidgets.QCheckBox(checked=True)
        hbox_gain.addWidget(autoGain)
        hbox_gain.addWidget(autoGainCheckBox)
        
        # reference to checkbox
        self.autoGainCheckBox = autoGainCheckBox
        
        hbox_fixedGain = QtWidgets.QHBoxLayout()
        fixedGain = QtWidgets.QLabel('Manual gain level for frequency spectrum')
        fixedGainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        hbox_fixedGain.addWidget(fixedGain)
        hbox_fixedGain.addWidget(fixedGainSlider)

        self.fixedGainSlider = fixedGainSlider


        #Empty Layout
        self.emptyHLayout = QtWidgets.QHBoxLayout()
        self.emptyHLayout.addWidget(QtWidgets.QLabel("\n"))



        self.SetButtons()
        self.SetUserInfo()
        self.SetTrainingLayout()
        self.UploadAudio()

        self.SetRecognitionLayout()

        #Assign layouts
        vbox = QtWidgets.QVBoxLayout()

        vbox.addLayout(self.user_info_layout)
        vbox.addLayout(self.hbox_Record)

        vbox.addLayout(self.emptyHLayout)
        vbox.addLayout(self.upload_audio_layout)
        vbox.addLayout(self.emptyHLayout)

        vbox.addLayout(self.trainingLayout)
        vbox.addLayout(hbox_gain)
    
        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)
        
        self.tab1.setLayout(vbox)
        self.tab2.setLayout(self.vbox2)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        
        self.setGeometry(200, 100, 600, 625)
        self.setWindowTitle('Speaker Recognition')
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setFixedSize(self.size())
        self.show()
        # timer for callbacks, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        timer.start(100)
        # keep reference to timer        
        self.timer = timer
       
    def setTabs(self):
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        
        #Add tabs to the tab widget
        self.tabs.addTab(self.tab1, "Enrollment")
        self.tabs.addTab(self.tab2, "Recognition")

    def SetRecognitionLayout(self):
        self.vbox2 = QtWidgets.QVBoxLayout()   
        
        recordLayout = QtWidgets.QHBoxLayout()
        Buttons = QtWidgets.QLabel('Record')
        record_button = QtWidgets.QPushButton('Record', self)
        record_button.setIcon(QtGui.QIcon("img/buttonIcon.png"))
        record_button.setToolTip('Record your voice.')
        self.progress = QtWidgets.QProgressBar()
        #progress.hide()
        completed_label = QtWidgets.QLabel("No audio")

        def onProgress(i):
            self.progress.setValue(i)
            if(i>99):
                completed_label.setText("Recording Complete")

        self.countWorker = CountWorker()
        self.countWorker.notifyProgress.connect(onProgress)

        @pyqtSlot()
        def on_record_button_click():
            completed_label.setText("Recording ...")
            worker = RecognitionWorker()
            LiveFFTWidget.threadpool.start(worker)
            self.countWorker.start()

        record_button.clicked.connect(on_record_button_click)
        recordLayout.addWidget(Buttons)
        recordLayout.addWidget(record_button)

        upload_button = QtWidgets.QPushButton('Upload', self)
        upload_button.setIcon(QtGui.QIcon("img/upload.png"))
        upload_button.setToolTip('Upload your voice.')
        @pyqtSlot()
        def on_upload_button_click(self):
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '/home')
            if fname:
                VarManager.recognitionAudioPath = fname
        
        upload_button.clicked.connect(on_upload_button_click)
        recordLayout.addWidget(upload_button)

        progressHLayout = QtWidgets.QHBoxLayout()
        progressHLayout.addWidget(self.progress)
        progressHLayout.addWidget(completed_label)


        recognizeLayout = QtWidgets.QHBoxLayout()

        recognizeButton = QtWidgets.QPushButton("Recognize", self)
        recognizeButton.setIcon(QtGui.QIcon("img/recognition.png"))
        recognizeButton.setToolTip('Recognize voice.')

        authenticateButton = QtWidgets.QPushButton("Authenticate")
        authenticateButton.setIcon(QtGui.QIcon("img/authentication.png"))
        authenticateButton.setToolTip('Record your voice.')

        recognizeButton.clicked.connect(self.on_recognize_button_click)
        authenticateButton.clicked.connect(self.on_authenticate_button_click)

        recognizeLayout.addWidget(recognizeButton)
        recognizeLayout.addWidget(authenticateButton)

        emptyLayout = QtWidgets.QHBoxLayout()
        emptyLayout.addWidget(QtWidgets.QLabel("\n"))
        self.vbox2.addLayout(emptyLayout)
        
        self.vbox2.addLayout(recordLayout)
        self.vbox2.addLayout(progressHLayout)
        self.vbox2.addLayout(recognizeLayout) 
        self.vbox2.addLayout(self.emptyHLayout)
        
        photoLabel = QLabel(self)
        pixmap = QtGui.QPixmap('img/audiobanner.jpg')
        pixmap = pixmap.scaled(556, 556, QtCore.Qt.KeepAspectRatio)
        photoLabel.setPixmap(pixmap)
        self.vbox2.addWidget(photoLabel)

    @pyqtSlot()
    def on_recognize_button_click(self):
        def completed(best_class):
            self.dialogBrowser = InfoDialog(self)
            self.dialogBrowser.displayInfo(best_class)
            self.dialogBrowser.exec_()
        self.recognitionThread = RecognitionThread()
        self.recognitionThread.recognizeCompleted.connect(completed)
        self.recognitionThread.start()

        recording_info = QtWidgets.QMessageBox.information(None, "Info!", "Recognizing audio...\n\nWait")



    @pyqtSlot()
    def on_authenticate_button_click(self):
        self.authenticateWindow = AuthenticateWindow(self)
        self.authenticateWindow.exec_()
        

    def SetTrainingLayout(self):
        self.trainingLayout = QtWidgets.QHBoxLayout()

        db_button = QtWidgets.QPushButton('Update Data', self)
        db_button.setIcon(QtGui.QIcon("img/dbIcon.png"))
        db_button.setToolTip('Update database')
        
        train_button = QtWidgets.QPushButton('Train', self)
        train_button.setIcon(QtGui.QIcon("img/trainingIcon.svg"))
        train_button.setToolTip("Train voice to neural network.")
        @pyqtSlot()
        def on_train_button_click():
            pass

        db_button.clicked.connect(self.on_db_button_click)
        train_button.clicked.connect(on_train_button_click)

        self.trainingLayout.addWidget(db_button)
        self.trainingLayout.addWidget(train_button)


    def on_db_button_click(self):
            
            VarManager.photoPath = "uploads/photos/photo_" 

            VarManager.userName = self.get_name()
            self.generateUserID()
            VarManager.photoPath += str(VarManager.userID) + VarManager.photoExt
            shutil.copy(VarManager.sourcePhotoPath, VarManager.photoPath)

            if(VarManager.audioUploaded):
                VarManager.audioPath = "uploads/audios/audio_"
                VarManager.audioPath += str(VarManager.userID) + ".wav"
                exists = os.path.isfile(VarManager.audioPath)
                if(exists):
                    pass
                else:
                    shutil.copy(VarManager.sourceAudioPath, VarManager.audioPath)
            audio_filename = os.path.basename(VarManager.audioPath)
            image_filename =  os.path.basename(VarManager.photoPath)
            multipart_form_data =  MultipartEncoder(
            fields={
            # a file upload field
            'audio': (audio_filename, open(VarManager.audioPath, 'rb')),
            'image':(image_filename, open(VarManager.photoPath, 'rb')),
            # plain text fields
            'uname': VarManager.userName, 
            'user_id':str(VarManager.userID),
           } )
            result = requests.post("http://127.0.0.1:5000/insert",data=multipart_form_data,headers={'Content-Type': multipart_form_data.content_type})
            data = result.json()
            if data["message"]=="Success":
                updateInfo = QtWidgets.QMessageBox.information(None, "Success", "Database Updated...")
            elif data["message"]=="Error":
                updateInfo = QtWidgets.QMessageBox.information(None, "Error", "Database Connection failed...")
            else:
                updateInfo = QtWidgets.QMessageBox.information(None, "Error", "Bad requests")

    def get_name(self):
        return self.nameEdit.text()

    def SetButtons(self):       #Record and Stop Audio buttons
        self.hbox_Record = QtWidgets.QHBoxLayout()
        Buttons = QtWidgets.QLabel('Record Audio:')
        record_button = QtWidgets.QPushButton('Record', self)
        record_button.setIcon(QtGui.QIcon("img/buttonIcon.png"))
        record_button.setToolTip('Record your voice.')
        mic = MicrophoneRecorder()
        @pyqtSlot()
        def on_record_button_click(self):
            if(LiveFFTWidget.recording==False):
                LiveFFTWidget.recording = True
                recording_info = QtWidgets.QMessageBox.information(None, "Info", "Recording started...\n\nClick Stop to stop recording")
                worker = Worker()
                LiveFFTWidget.threadpool.start(worker)
            else:
                recording_info = QtWidgets.QMessageBox.information(None, "Warning!", "Already Recording audio...\n\nClick Stop to stop recording")

        record_button.clicked.connect(on_record_button_click)
        self.hbox_Record.addWidget(Buttons)
        self.hbox_Record.addWidget(record_button)

        stop_button = QtWidgets.QPushButton('Stop', self)
        stop_button.setToolTip('Stop Recording.')
        stop_button.setIcon(QtGui.QIcon("img/stopIcon.png"))
        @pyqtSlot()
        def on_stop_button_click(self):
            if(LiveFFTWidget.recording==True):
                mic.stopRecord = True
                LiveFFTWidget.stopRecord = 1
                LiveFFTWidget.recording = False
                recording_info = QtWidgets.QMessageBox.information(None, "Info", "Recording complete...")
            else:
                recording_info = QtWidgets.QMessageBox.information(None, "Cannot Stop", "Recording not started!")

        stop_button.clicked.connect(on_stop_button_click)
        self.hbox_Record.addWidget(stop_button)

        self.stopButton = stop_button

    def UploadAudio(self):
        self.upload_audio_layout = QtWidgets.QHBoxLayout()

        upload_button = QtWidgets.QPushButton('Upload Audio', self)
        upload_button.setIcon(QtGui.QIcon("img/upload.png"))
        upload_button.setToolTip('Upload your audio')
        textEdit = QtWidgets.QLabel()
        @pyqtSlot()
        def on_upload_button_clicked(self):
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '/home')
            if fname:
                destName = 'uploads/audios/' + "audio_"
                textEdit.setText(fname)
                VarManager.sourceAudioPath = fname
                VarManager.audioPath = destName
                VarManager.audioUploaded = True
        upload_button.clicked.connect(on_upload_button_clicked)
        self.upload_audio_layout.addWidget(upload_button)
        self.upload_audio_layout.addWidget(textEdit)

    def SetUserInfo(self):
        self.user_info_layout = QtWidgets.QHBoxLayout()
        nameLabel = QtWidgets.QLabel("Name: ")
        self.nameEdit = QtWidgets.QLineEdit()
        
        upload_button = QtWidgets.QPushButton('Upload Photo', self)
        upload_button.setIcon(QtGui.QIcon("img/pic.png"))
        upload_button.setToolTip('Upload your photo')
        textEdit = QtWidgets.QLabel()
        @pyqtSlot()
        def on_upload_button_clicked(self):
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '/home')
            if fname:
                destName = 'uploads/photos/' + "photo_"

                textEdit.setText(fname)
                VarManager.sourcePhotoPath = fname
                VarManager.photoPath = destName

                VarManager.photoExt = pathlib.Path(fname).suffix
        upload_button.clicked.connect(on_upload_button_clicked)
        self.user_info_layout.addWidget(nameLabel)
        self.user_info_layout.addWidget(self.nameEdit)
        self.user_info_layout.addWidget(upload_button)
        self.user_info_layout.addWidget(textEdit)
     
    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()  

        # keeps reference to mic        
        self.mic = mic
        
        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize, 
                                         1./mic.rate)
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
                
    def connectSlots(self):
        pass
    
    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps 
        references for further use"""
        # top plot
        self.ax_top = self.main_figure.figure.add_subplot(211)
        self.ax_top.set_ylim(-5000, 5000)
        self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)
        self.ax_top.set_ylabel(u'amp', fontsize=6)
        # bottom plot
        self.ax_bottom = self.main_figure.figure.add_subplot(212)
        self.ax_bottom.set_ylim(0, 1)
        self.ax_bottom.set_xlim(0, 5000)
        self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)
        self.ax_bottom.set_ylabel(u'amp', fontsize=6)
        
        # line objects
        self.main_figure.figure.tight_layout()        
        self.line_top, = self.ax_top.plot(self.time_vect, 
                                         np.ones_like(self.time_vect))

        self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                               np.ones_like(self.freq_vect))
                                    
    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """        
        # gets the latest frames        
        frames = self.mic.get_frames()
        
        if len(frames) > 0:
            # keeps only the last frame
            current_frame = frames[-1]
            # plots the time signal
            self.line_top.set_data(self.time_vect, current_frame)
            # computes and plots the fft signal            
            fft_frame = np.fft.rfft(current_frame)
            if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                fft_frame /= np.abs(fft_frame).max()
            else:
                fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
            self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))            
            # refreshes the plots
            self.main_figure.canvas.draw()

'''
Recognition thread
'''
class RecognitionThread(QtCore.QThread):
    recognizeCompleted = QtCore.pyqtSignal(dict)

    @pyqtSlot()
    def run(self):
        '''
        Recognition
        '''
        best_class= self.predict()
        self.recognizeCompleted.emit(best_class)

    # predicting the user
    def predict(self):
        file_path = VarManager.recognitionAudioPath 
        audio_filename = os.path.basename(file_path) 
        multipart_form_data = {
        'file': (audio_filename, open(file_path, 'rb'))
        }
    
        result = requests.post("http://127.0.0.1:5000/identification",files=multipart_form_data)
        data = result.json() #convert result into json
        #Make api request by post/get sending audio path
        return data



