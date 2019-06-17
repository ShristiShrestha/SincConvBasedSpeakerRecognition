import pyaudio
import threading
import atexit
import numpy as np
from datetime import datetime
import wave

class MicrophoneRecorder(object):

	stopRecord = False;

	def __init__(self, rate=44100, chunksize=1024):

		self.rate = rate
		self.chunksize = chunksize
		self.channels = 1
		self.format = pyaudio.paInt16

		self.recordseconds = 10

		self.p = pyaudio.PyAudio()

		# self.stream = self.p.open(format=pyaudio.paInt16,
		# 	channels=1,
		# 	rate=self.rate,
		# 	input=True,
		# 	frames_per_buffer=self.chunksize,
		# 	stream_callback=self.new_frame)

		self.stream = self.p.open(format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunksize,
                stream_callback=self.new_frame)

		self.lock = threading.Lock()
		self.stop = False
		self.frames = []
		atexit.register(self.close)


	def new_frame(self, data, frame_count, time_info, status):
		data = np.fromstring(data, 'int16')
		with self.lock:
			self.frames.append(data)
			if self.stop:
				return None, pyaudio.paComplete
		return None, pyaudio.paContinue


	def get_frames(self):
		with self.lock:
			frames = self.frames
			self.frames = []
			return frames

	def start(self):
		self.stream.start_stream()

	def close(self):
		with self.lock:
			self.stop = True
		self.stream.close()
		self.p.terminate()


	# def record(self):
	# 	x = threading.Thread(target=self.recordAudio)
	# 	x.start()
	# 	x.join()


	def record(self):
		datetime_string = datetime.now().strftime("%d%m%Y_%H%M%S")
		filename = "record_"+datetime_string+".wav"
		
		if(self.p.get_host_api_info_by_index(0).get('deviceCount') < 1):
			print("No recording device found")
			return

		stream = self.p.open(format=self.format,
			rate=self.rate,
			channels=self.channels,
			input=True,
			frames_per_buffer=self.chunksize)
		print("Recording...")

		Recordframes = []

		for i in range(0, int(self.rate/self.chunksize * self.recordseconds)):
			data = stream.read(self.chunksize)
			Recordframes.append(data)
			if(self.stopRecord):
				self.stopRecord = False
				break
		print("Recording Stopped.")

		stream.stop_stream()
		stream.close()

		self.saveFile(filename, Recordframes)

	def saveFile(self, filename, recordframes):
		waveFile = wave.open(filename, 'wb')
		waveFile.setnchannels(self.channels)
		waveFile.setsampwidth(self.p.get_sample_size(self.format))
		waveFile.setframerate(self.rate)
		waveFile.writeframes(b''.join(recordframes))
		waveFile.close()

