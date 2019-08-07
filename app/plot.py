# plot the training loss and accuracy
import numpy as np
from matplotlib import pyplot as plt
import csv
from sklearn.metrics import classification_report,precision_score
import soundfile as sf
"""a = []
b =[]
c = []
d =[]
a1 = []
b1 =[]
c1 = []
d1 =[]

N = 1
data_file =["trial_1.csv","trial_2.csv"."trial_3.csv"]

for x in data_file:
with open('trial3_classification.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if N<=201:
            a.append(float(row[2]))#tacc
            b.append(float(row[3]))#tlos
            c.append(float(row[4]))#vacc
            d.append(float(row[5]))
            N=N+1
N=1
with open('cnn_performance_librispeech.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if N<=201:
            a1.append(float(row[2]))#tacc
            b1.append(float(row[3]))#tlos
            c1.append(float(row[4]))#vacc
            d1.append(float(row[5]))
            N = N+1
       

N = 201
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), b,'-b', label="cnn_train_loss")
plt.plot(np.arange(0,N), b1,'-r', label="sinc_train_loss")
plt.title("Comparing CNN and SincNet for librispeech dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("loss.png")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), a,'-b', label="cnn_train_acc")
plt.plot(np.arange(0,N), a1,'-r', label="sinc_train_acc")
plt.title("Comparing CNN and SincNet for librispeech dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("acc.png")
"""
"""N=221
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), a,'-b', label="train_acc")
plt.plot(np.arange(0, N), c,'-r', label="val_acc")
plt.title(" Training/Validation accuracy for 265 speakers")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("acc.png")"""

from tqdm import tqdm
import soundfile as sf
import numpy as np
import pandas as pd
from model import get_model
import soundfile as sf
import tensorflow as tf
import seaborn as sns
from keras import backend as K
from Configuration import Batch_dev, data_folder, lab_dict, wav_lst_te, wav_lst_tr, wlen,wshift, class_lay, seed,N_eval_epoch,lowest_id, output_folder,pt_file,fs
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
import matplotlib
import csv
np.random.seed(seed)

class Validation():
    def __init__(self, Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay, model, debug=False):
        self.wav_lst_te = wav_lst_te
        self.data_folder = data_folder
        self.wlen = wlen
        self.wshift = wshift
        self.lab_dict = lab_dict
        self.Batch_dev = Batch_dev
        self.class_lay = class_lay
        self.model = model
        self.debug = debug
        self.pred_class  = []
        self.actual_class = []
        
    def plot_confusion_matrix(self,y_true, y_pred, classes,normalize=False,title=None,cmap=None):
        
        if not title:
            if normalize:
                    title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        print(classification_report(y_true,y_pred))
        return ax

    def validate(self, epoch=None):
        data_folder = self.data_folder
        wav_lst_te = self.wav_lst_te
        wlen = self.wlen
        wshift = self.wshift
        lab_dict = self.lab_dict
        Batch_dev = self.Batch_dev
        class_lay = self. class_lay
        debug = self.debug
        
        if epoch==None:
           
            print('Valuating test set...')
            

            snt_te=len(wav_lst_te)
            #aadded later

            err_sum = 0
            err_sum_snt = 0
            stn_sum = 0
            if debug:
                print('WLEN: '+str(wlen))
                print('WSHIFT: '+str(wshift))
                pbar = tqdm(total=snt_te)
            for i in range(snt_te):
                [signal, fs] = sf.read(data_folder+wav_lst_te[i])

                #signal = tf.convert_to_tensor(signal, dtype=float)
                signal = np.array(signal)
                lab_batch=lab_dict[data_folder + wav_lst_te[i]]

                #split signals into chunck
                beg_samp=0
                end_samp=wlen

                N_fr=int((signal.shape[0]-wlen)/(wshift))

                #sig_arr=K.zeros([Batch_dev,wlen],dtype=float)
                sig_arr=np.zeros([Batch_dev,wlen])
                lab=np.zeros(N_fr+1)+lab_batch
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
                        #inp = K.variable(sig_arr)
                        pout[count_fr_tot-Batch_dev:count_fr_tot,:] = self.model.predict(inp, verbose=0)
                        count_fr=0
                        sig_arr=np.zeros([Batch_dev,wlen])

                    #Add the last items left 
                if count_fr>0:
                    inp = sig_arr[0:count_fr]
                    a,b = np.shape(inp)
                    inp = np.reshape(inp,(a,b,1))
                    #inp = np.array(inp)
                    pout[count_fr_tot-count_fr:count_fr_tot,:] = self.model.predict(inp, verbose=0)

                #Prediction for each chunkc  and calculation of average error
                pred = np.argmax(pout, axis=1)
                err = np.mean(pred!=lab)
                #pred =K.argmax(pout, axis=1)
                #err = K.mean(pred!=lab)
                
                #Calculate accuracy on the whole sentence
                best_class = np.argmax(np.sum(pout, axis=0))
                #best_class = K.argmax(K.sum(pout, axis=0))
                with open("mid_defense_output.csv","a") as f:
                    writer=csv.writer(f)
                    writer.writerow([best_class,lab_batch])
                if best_class>=252 and best_class<=265:

                    self.pred_class.append(best_class)
                    self.actual_class.append(lab_batch)

                err_sum_snt = err_sum_snt+float((best_class!=lab[0]))
                err_sum = err_sum + err

                stn_sum += 1

                temp_acc_stn = str(round(1-(err_sum_snt/stn_sum), 4))
                temp_acc = str(round(1-(err_sum/stn_sum), 4))
                if debug:
                    pbar.set_description('acc: {}, acc_snt: {}'.format(temp_acc, temp_acc_stn))
                    pbar.update(1)

            #average accuracy
            acc = 1-(err_sum/snt_te)
            acc_snt = 1-(err_sum_snt/snt_te)
            if debug:
                pbar.close()
            if epoch is None:
                print('acc_te: {}, acc_te_snt: {}\n'.format(acc, acc_snt))
                
            else:
                print('Epoch: {}, acc_te: {}, acc_te_snt: {}\n'.format(epoch, acc, acc_snt))
                with open(output_folder+"/res.res", "a") as res_file:
                    res_file.write("epoch %i, acc_te=%f acc_te_snt=%f\n" % (epoch, acc, acc_snt)) 
            np.set_printoptions(precision=2)
            class_names = np.arange(252,265)
            # Plot non-normalized confusion matrix
            self.plot_confusion_matrix(self.actual_class,self.pred_class, classes=class_names,title='Confusion matrix, without normalization')

            plt.show()
            plt.savefig("conf_matrix.png")  
            return (acc, acc_snt)
            

def main():
    print("Validation...")
    if pt_file!='none':
        weight_file = pt_file
        input_shape = (wlen, 1)
        out_dim = class_lay[0]
        model = get_model(input_shape, out_dim)
        model.load_weights(weight_file)
        val = Validation(Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay, model, True)
        print(val.validate())
        
       
    else:
        
        print("No PT FILE")
    
def plot():
    a =[]
    b = []

    with open('trial1_classificagtion.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
        
            a.append(int(row[0]))#predicted class
            b.append(int(row[1]))#actual class
            
   
    # Compute confusion matrix
    cm = confusion_matrix(b, a)
    # Creates a confusion matrix
    cm = cm[251:264, 251:264]
    # Transform to df for easier plotting
    cm_df = pd.DataFrame(np.array(cm),
                     index = np.arange(1, 14), 
                     columns = np.arange(1, 14))
    print(cm_df)
    plt.figure(figsize=(6.5,6))
    sns.heatmap(cm_df, annot=True)
    plt.title('SincNet Model \nAccuracy:{0:.3f}'.format(accuracy_score(b, a))+'\nF1 Score:{0:.3f}'.format(f1_score(b, a, average='micro'))+'\nPrecision:{0:.3f}'.format(precision_score(b, a, average='micro')))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('trial1_classfication_matrix.png')

    plt.show()
if __name__ == "__main__":
    
    plot()
