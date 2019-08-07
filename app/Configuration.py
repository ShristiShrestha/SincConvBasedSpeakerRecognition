from optparse import OptionParser
import configparser as cg


def read_list(filename):
    file = open(filename,'r')
    lines = file.readlines()
    list_signal = []
    for x in lines:
        list_signal.append(x.rstrip())
    file.close()
    return list_signal


def read_config(cfg_file = None):
    parser = OptionParser()
    parser.add_option("--cfg") #Compulsory
    (options, _args) = parser.parse_args()
    if cfg_file is None:
        cfg_file = options.cfg
    if options.cfg is None and cfg_file is None:
        cfg_file = 'configuration/sincnet_nikita.cfg'
    Config = cg.ConfigParser()
    Config.read(cfg_file)

    #[data]
    options.tr_lst = Config.get('data','tr_lst')
    options.te_lst = Config.get('data', 'te_lst')
    options.lab_dict = Config.get('data', 'lab_dict')
    options.data_folder = Config.get('data', 'data_folder')
    options.output_folder = Config.get('data', 'output_folder')
    options.pt_file = Config.get('data', 'pt_file')
    options.lowest_id = Config.get('data', 'lowest_id')

    #[windowing]
    options.fs = Config.get('windowing', 'fs')
    options.cw_len = Config.get('windowing', 'cw_len')
    options.cw_shift = Config.get('windowing', 'cw_shift')

    #[cnn]
    options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = Config.get('cnn', 'cnn_act')
    options.cnn_drop = Config.get('cnn', 'cnn_drop')

    #[dnn]
    options.fc_lay = Config.get('dnn', 'fc_lay')
    options.fc_drop = Config.get('dnn', 'fc_drop')
    options.fc_use_laynorm_inp = Config.get('dnn', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp = Config.get('dnn', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm = Config.get('dnn', 'fc_use_batchnorm')
    options.fc_use_laynorm = Config.get('dnn', 'fc_use_laynorm')
    options.fc_act = Config.get('dnn', 'fc_act')

    #[classes]
    options.class_lay = Config.get('class', 'class_lay')
    options.class_drop = Config.get('class', 'class_drop')
    options.class_use_laynorm_inp = Config.get('class', 'class_use_laynorm_inp')
    options.class_use_batchnorm_inp = Config.get('class', 'class_use_batchnorm_inp')
    options.class_use_batchnorm = Config.get('class', 'class_use_batchnorm')
    options.class_use_laynorm = Config.get('class', 'class_use_laynorm')
    options.class_act = Config.get('class', 'class_act')

    #[optimization]
    options.lr = Config.get('optimization', 'lr')
    options.batch_size = Config.get('optimization', 'batch_size')
    options.N_epochs = Config.get('optimization', 'N_epochs')
    options.N_batches = Config.get('optimization', 'N_batches')
    options.N_eval_epoch = Config.get('optimization', 'N_eval_epoch')
    options.seed = Config.get('optimization', 'seed')
    return options


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError

import numpy as np


#Reading cfg file

options = read_config("trial_1.cfg")


#[data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder+'/'
output_folder = options.output_folder
lowest_id = 0

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)

# training list
wav_lst_tr=read_list(tr_lst) #Elenco file audio per il train
snt_tr=len(wav_lst_tr)

# test list
wav_lst_te=read_list(te_lst)
snt_te=len(wav_lst_te)


#converting context and shift in shamples
wlen = int(fs*cw_len/1000.00)
wshift = int(fs*cw_shift/1000.00)

#Batch_dev
Batch_dev =128

#loading label dictionary
lab_dict = np.load(class_dict_file).item()


#initialiation of minibatch (batch_size,[0=>x_t ,1=>x_t+N,1=>Random_Sample])

output_dimension = class_lay[0]