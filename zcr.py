import numpy as np, soundfile as sf, math, json, librosa as lb
import csv, os
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