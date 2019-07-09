from model import get_model
from Configuration import *
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

import  math, json, librosa as lb
import csv
import scipy.signal

class ZeroCrossing:
    def __init__(self):

        pass
    
    def Load(self, filepath):                   
        self.data, self.sampling_rate= lb.load(filepath, sr=16000)
        
        return
	
    def Label(self):
        self.np_data=np.zeros(len(self.data))
        self.signal={}
        for i in range(0, len(self.data)):
            self.np_data[i]=np.mean(self.data[i])
            self.signal[i]={}
            self.signal[i]['Value']=self.np_data[i]
            if self.np_data[i]>=0:
                self.signal[i]['Label']=1
            else:
                self.signal[i]['Label']=0
        
        return 

    def Calc(self):
        self.frame_size=int(self.sampling_rate*0.01) #10ms frame
        self.num_frame=math.floor(len(self.data)/self.frame_size)
        self.frame=np.zeros(shape=(self.num_frame, 3)) #third col: 0 for unvoiced frame
        window = scipy.signal.get_window("hamming", self.frame_size)
        
        for i in range(0, self.num_frame):
            self.frame[i][0]=sum([abs(self.signal[(self.frame_size*i)+j]['Label']-self.signal[(self.frame_size*i)+j-1]['Label']) for j in range(1,self.frame_size)])#ZCR 
            self.frame[i][1]=sum([(self.np_data[(self.frame_size*i)+j]*window[j])**2 for j in range(self.frame_size)]) #Energy 
        
        
        self.thr=self.EnergyThreshold()   
        
        for i in range(0, self.num_frame):
            if self.frame[i][0]<=50 and self.frame[i][1]>=self.thr[0] and self.frame[i][1]<=self.thr[2]: #low ZCR z<=1 and high Energy
                self.frame[i][2]=1 #Voiced  

        return
    
    def Write(self,fname, dest_fname):
        self.Load(fname)
        self.Label()
        self.Calc()
        data=[]
        for i in range(0, self.num_frame):
            if self.frame[i][2]==1: #voiced
               # print("%d ko %d " %(i, self.frame[i][2])) 
                for j in range(0, self.frame_size):
                    data=np.append(data,self.np_data[(self.frame_size*i)+j])           
                          
        c=len(self.data)-(self.num_frame*self.frame_size)
        
        for i in range(c): #add last dtpts
            data=np.append(data, self.np_data[(self.num_frame*self.frame_size)+i])
        

        if len(data)>0:
            
            lb.output.write_wav(dest_fname, np.array(data), self.sampling_rate)
            
        else:
            print("Zero rm_data pts", fname)
        
        return data

    def EnergyThreshold(self):
        Emax=max([self.frame[i][1] for i in range(self.num_frame)])
        Emin=min([self.frame[i][1] for i in range(self.num_frame)])
        if Emin==0.0:
            Emin=0.00001
        T1=Emin*(1+2*math.log10(Emax/Emin))
        SL=sum([self.frame[i][1] for i in range(self.num_frame) if self.frame[i][1]>T1])/sum([1 for i in range(self.num_frame) if self.frame[i][1]>T1])
        T2=T1+0.75*(SL-T1)
        
        return [T1, SL, T2]

def NoiseRemoval(fsrc, fdest):

    y, sr = librosa.load(fsrc)
    S_full, phase = librosa.magphase(librosa.stft(y)) 
    idx = slice(*librosa.time_to_frames([0,7], sr=sr))
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)
    
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    

    x = librosa.istft(S_foreground) #final data to ZCR 
    x=x*2
    #librosa.output.write_wav(fdest,x, sr)

    return x

def test(audio_path):
    weight_file = pt_file
    input_shape = (wlen,1)
    out_dim = 262
    model = get_model(input_shape, out_dim)
    model.load_weights(weight_file)
    [signal, fs] = sf.read(audio_path)
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
    
    #Calculate accuracy on the whole sentence
    best_class = np.argmax(np.sum(pout, axis=0))
    
    return best_class


"""i = 0
    for layer in model.layers:
        #layer.trainable = False
        i = i+1
        print(i,layer.name)

"""