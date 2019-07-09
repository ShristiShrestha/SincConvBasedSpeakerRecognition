import numpy as np
import librosa
import soundfile as sf
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_path = "data"

def wav2mfcc(file_path, max_pad_len=11):
	wave, sr = sf.read(file_path)
	wave = wave[::3]
	mfcc = librosa.feature.mfcc(wave, sr=16000)

	if(max_pad_len > mfcc.shape[1]):
		pad_width = max_pad_len - mfcc.shape[1]
		mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
	else:
		mfcc = mfcc[:, :max_pad_len]
	return mfcc


def getlabel(path = data_path):
	labels = os.listdir(path)
	label_indices = np.arange(0, len(labels))
	return labels, label_indices, to_categorical(label_indices)


def save_mfcc_toarray(path = data_path, max_pad = 11):
	labels, _,_ = getlabel(path)
	for label in labels:
		mfcc_vectors = []

		wavfiles = [path +'/'+ label + '/' + wavfile for wavfile in os.listdir(path+'/'+label)]
		for wav in wavfiles:
			mfcc = wav2mfcc(wav)
			mfcc_vectors.append(mfcc)
		np.save(label+'.npy',mfcc_vectors)
		print("Succes")


def get_train_test(split_ratio = 0.6, random_state = 42):
	#Get available labels
	labels, indices, _ = getlabel(data_path)

	#Getting first arrays from labels
	X = np.load(labels[0] + '.npy')
	y = np.zeros(X.shape[0])

	#Append all the dataset into one signle array
	for i, label in enumerate(labels[1:]):
		x = np.load(label + '.npy')
		X = np.vstack((X, x))
		y = np.append(y, np.full(x.shape[0], fill_value=(i+1)))

	assert X.shape[0] == len(y)
	return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU

X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


#Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20,11,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20,11,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=keras.optimizers.Adadelta(),
	metrics=['accuracy'])

model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))







sample = wav2mfcc("data/9/115.flac")
sample2 = wav2mfcc("data/1/100.flac")
sample_reshaped = sample.reshape(1,20,11,1)
sample2_reshaped = sample2.reshape(1, 20, 11, 1)

print(getlabel()[0][
	np.argmax(model.predict(sample_reshaped))])

print(getlabel()[0][
	np.argmax(model.predict(sample2_reshaped))])
