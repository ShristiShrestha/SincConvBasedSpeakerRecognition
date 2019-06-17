import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from datetime import datetime
import time
import shutil
from model import get_model
import os
import soundfile as sf
from Configuration import *
from test_nikita import Validation
from MicRecorder import MicrophoneRecorder
from Interface import MplFigure, CreateFolder
from Recorder import Recorder
from database import *
from RecognitionTab import *
from VarManager import VarManager


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
        print("Thread start") 
        micRecorder = Recorder(channels=1)

        datetime_string = datetime_to_string()
        filename = "record_"+datetime_string+".wav"

        with micRecorder.open("records/"+filename, 'wb') as recfile:
            recfile.start_recording()
            #time.sleep(10.0)
            while (LiveFFTWidget.stopRecord!=1):
                pass    
            recfile.stop_recording()
            LiveFFTWidget.stopRecord = 0

        VarManager.audioPath = "records/" + filename
        VarManager.audioUploaded = False
        print(VarManager.audioPath)

        print("Recording complete")


class RecognitionWorker(QRunnable):
    '''
    Worker thread
    '''
    @pyqtSlot()
    def run(self):
        '''
        record
        '''
        print("Thread start") 
        micRecorder = Recorder(channels=1)

        datetime_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = "record_"+datetime_string+".wav"
        with micRecorder.open("records/testing/"+filename, 'wb') as recfile:
            recfile.start_recording()
            time.sleep(10.0)    
            recfile.stop_recording()
            LiveFFTWidget.stopRecord = 0

        print("Recording complete")


class LiveFFTWidget(QWidget):

    stopRecord = 0
    recording = False
    threadpool = QThreadPool()

    def __init__(self):

        QWidget.__init__(self)

        self.iconName = "img/appIcon.svg"
        CreateFolder("records")

        self.uploadPhotoName = ""
        self.file_nname = ""
        
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

        self.setTabs();

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


        self.SetButtons()
        self.SetUserInfo()
        self.SetTrainingLayout()
        self.UploadAudio()

        self.SetRecognitionLayout()


        #Empty Layout
        self.emptyHLayout = QtWidgets.QHBoxLayout()
        self.emptyHLayout.addWidget(QtWidgets.QLabel("\n"))

        #Assign layouts
        vbox = QtWidgets.QVBoxLayout()

        vbox.addLayout(self.user_info_layout)
        vbox.addLayout(self.hbox_Record)

        vbox.addLayout(self.emptyHLayout)
        vbox.addLayout(self.upload_audio_layout)
        vbox.addLayout(self.emptyHLayout)

        vbox.addLayout(self.trainingLayout)
        vbox.addLayout(hbox_gain)
        #vbox.addLayout(hbox_fixedGain)
        

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)


        #RecognitionTab
        # self.recognition = Recognition(self)
        # self.vbox2.addWidget(self.recognition)
        #self.dialogTextBrowser = MyDialog(self)


        
        self.tab1.setLayout(vbox)
        self.tab2.setLayout(self.vbox2)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        
        self.setGeometry(200, 200, 600, 500)
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
        #self.tabs.resize(300, 200)
        
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
        progress = QtWidgets.QProgressBar()
        @pyqtSlot()
        def on_record_button_click(self):
            worker = RecognitionWorker()
            LiveFFTWidget.threadpool.start(worker)
            count = 0
            while count < 20:
                count+=1
                time.sleep(0.5)
                progress.setValue(count*5)
        
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
                #destName = 'uploads/audios/' + "audio_"

                # textEdit.setText(fname)
                # VarManager.sourceAudioPath = fname
                # VarManager.audioPath = destName   
                # VarManager.audioUploaded = True
                VarManager.recognitionAudioPath = fname
                print(fname)
        
        upload_button.clicked.connect(on_upload_button_click)
        recordLayout.addWidget(upload_button)

        progressHLayout = QtWidgets.QHBoxLayout()
        progressHLayout.addWidget(progress)


        recognizeLayout = QtWidgets.QHBoxLayout()
        recognizeButton = QtWidgets.QPushButton("Recognize", self)


        recognizeButton.clicked.connect(self.on_recognize_button_click)
        recognizeLayout.addWidget(recognizeButton)

        self.vbox2.addLayout(recordLayout)
        self.vbox2.addLayout(progressHLayout)
        self.vbox2.addLayout(recognizeLayout) 

    @pyqtSlot()
    def on_recognize_button_click(self): #this is for reognising 
        weight_file ="./output_nikitaa/checkpoints/SincNet.hdf5"
        input_shape = (wlen, 1)
        out_dim = class_lay[0]
        model = get_model(input_shape, out_dim)
        model.load_weights(weight_file)
        x = VarManager.recognitionAudioPath
        
        [signal, fs] = sf.read(VarManager.recognitionAudioPath)
        signal = np.array(signal)
        #lab_batch=lab_dict[data_folder + wav_lst_te[i]]
        #split signals into chunck
        beg_samp=0
        end_samp=wlen
        N_fr=int((signal.shape[0]-wlen)/(wshift))
        sig_arr=np.zeros([Batch_dev,wlen])
        pout =np.zeros(shape=(N_fr+1,class_lay[-1]))
        count_fr=0
        count_fr_tot=0
        while end_samp<signal.shape[0]: #for each chunck
            sig_arr[count_fr,:]=signal[beg_samp:end_samp]
            beg_samp=beg_samp+wshift
            end_samp=beg_samp+wlen
            count_fr=count_fr+1
            count_fr_tot=count_fr_tot+1
            if count_fr==Batch_dev: 
                a,b = np.shape(sig_arr)
                inp = sig_arr.reshape(a,b,1)
                inp = np.array(inp)
                pout[count_fr_tot-Batch_dev:count_fr_tot,:] = model.predict(inp, verbose=0)
                count_fr=0
                sig_arr=np.zeros([Batch_dev,wlen])
            #Add the last items left 
        if count_fr>0:
            inp = sig_arr[0:count_fr]
            a,b = np.shape(inp)
            inp = np.reshape(inp,(a,b,1))
            pout[count_fr_tot-count_fr:count_fr_tot,:] = model.predict(inp, verbose=0)
        #Prediction for each chunkc  and calculation of average error
        pred = np.argmax(pout, axis=1)
        best_class = np.argmax(np.sum(pout, axis=0))
        print(best_class)
        print(VarManager.recognitionAudioPath)
        """self.dialogTextBrowser = InfoDialog(self)
        self.dialogTextBrowser.exec_()"""

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
            VarManager.audioPath = "uploads/audios/audio_"    

            VarManager.userName = self.get_name()
            self.generateUserID()
            VarManager.photoPath += str(VarManager.userID) + ".png"
            shutil.copy(VarManager.sourcePhotoPath, VarManager.photoPath)

            if(VarManager.audioUploaded):
                VarManager.audioPath += str(VarManager.userID) + ".wav"
                exists = os.path.isfile(VarManager.audioPath)
                if(exists):
                    pass
                else:
                    shutil.copy(VarManager.sourceAudioPath, VarManager.audioPath)

            connect_to_db(VarManager.userName, VarManager.userID, VarManager.audioPath, VarManager.photoPath)
            updateInfo = QtWidgets.QMessageBox.information(None, "Success", "Database Updated...")

    #get_name = lambda x: self.nameEdit.text()
    def get_name(self):
        return self.nameEdit.text()

    def SetButtons(self):
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
        #textEdit = QtWidgets.QTextEdit()
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
                print(destName)

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
        #textEdit = QtWidgets.QTextEdit()
        textEdit = QtWidgets.QLabel()
        @pyqtSlot()
        def on_upload_button_clicked(self):
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '/home')
            if fname:
                destName = 'uploads/photos/' + "photo_"

                textEdit.setText(fname)
                VarManager.sourcePhotoPath = fname
                VarManager.photoPath = destName
                print(destName)

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
        #self.ax_top.set_ylim(-32768, 32768)
        self.ax_top.set_ylim(-5000, 5000)
        self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)
        self.ax_top.set_ylabel(u'amp', fontsize=6)
        

        # bottom plot
        self.ax_bottom = self.main_figure.figure.add_subplot(212)
        self.ax_bottom.set_ylim(0, 1)
        #self.ax_bottom.set_xlim(0, self.freq_vect.max())
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
                #print(np.abs(fft_frame).max())
            self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))            
            
            # refreshes the plots
            self.main_figure.canvas.draw()
