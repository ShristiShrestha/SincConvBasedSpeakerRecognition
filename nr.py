import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display
import os

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
