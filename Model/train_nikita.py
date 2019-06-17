import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.optimizers import RMSprop,Adam
from keras import backend as K
from model import get_model

from test_nikita import Validation
import csv, time
from Configuration import Batch_dev,snt_te, data_folder, lab_dict, wav_lst_te, wav_lst_tr, wlen,wshift, class_lay, seed,N_eval_epoch,lowest_id, output_folder,pt_file,fs, lr, cnn_N_filt, cnn_len_filt, batch_size,N_batches,N_epochs,snt_tr
import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt

debug = False
K.clear_session()

def debug_print(*objects):
    if debug:
        print(objects)
        
def batchGenerator(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, out_dim):
    while True:
        sig_batch, lab_batch = create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, out_dim)
        yield sig_batch, lab_batch

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, out_dim):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
    lab_batch=[]
    #lab_batch=np.zeros(batch_size)
    snt_id_arr=np.random.randint(N_snt, size=batch_size)
    #debug_print("Sentence_id_array: ",snt_id_arr )
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
    for i in range(batch_size): 
        # select a random sentence from the list 
        [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]])
        # accesing to a random chunk
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-wlen-1) 
        snt_end=snt_beg+wlen
        sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
        y=lab_dict[data_folder+wav_lst[snt_id_arr[i]]]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
  
    return sig_batch, np.array(lab_batch)
    

class ValidationCallback(Callback):
    def __init__(self, Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay):
        self.wav_lst_te = wav_lst_te
        self.data_folder = data_folder
        self.wlen = wlen
        self.wshift = wshift
        self.lab_dict = lab_dict
        self.Batch_dev = Batch_dev
        self.class_lay = class_lay
    def on_epoch_end(self, epoch, logs={}):
        val = Validation(self.Batch_dev, self.data_folder, self.lab_dict, self.wav_lst_te, self.wlen, self.wshift, self.class_lay, self.model)
        val.validate(epoch)



"""print('N_filt '+str(cnn_N_filt))
print('N_filt len '+str(cnn_len_filt))
print('FS '+str(fs))
print('WLEN '+str(wlen))"""

input_shape = (wlen, 1)
out_dim = class_lay[0]

model = get_model(input_shape, out_dim)
optimizer1=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
optimizer = RMSprop(lr=lr, rho=0.95, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])



checkpoints_path = os.path.join(output_folder,'checkpoints')

tb = TensorBoard(log_dir=os.path.join(output_folder,'logs', 'SincNet'))
checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path,  'SincNet.hdf5'),
        verbose=1,
        save_best_only=False)

if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)


validation = ValidationCallback(Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay)
callbacks = [tb,checkpointer]


if pt_file!='none':
   model.load_weights(pt_file)

train_generator = batchGenerator(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2, out_dim)
test_generator = batchGenerator(batch_size,data_folder,wav_lst_te,snt_te,wlen,lab_dict,0.2, out_dim)
history = model.fit_generator(train_generator, steps_per_epoch=N_batches, epochs=N_epochs, verbose=1, callbacks=callbacks, validation_data=test_generator, validation_steps = 5)
#x = model.evaluate_generator(test_generator,)
print(history.history)

ts=time.gmtime()
timestamp=str(time.strftime("%Y-%m-%d %H:%M:%S", ts))

write_csv=open('Performance_less_1.csv', 'a')
writer=csv.writer(write_csv)

title=["timestamp","learning_rate","N_epochs", "N_batches","Batch_Size","N_eval_epoch","acc", "loss"]
writer.writerow(title)

acc=np.array(history.history['acc'])
loss=np.array(history.history['loss'])
data=zip(*[acc, loss])
for row in data:
    writer.writerow([i] for i in row)
write_csv.close()
for row in data:
    content=[timestamp,lr, batch_size, N_epochs, N_batches, N_eval_epoch, row[0], row[1] ]
    writer.writerow(content)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()