from model import get_model
from Configuration import *
import soundfile as sf
import numpy as np

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


