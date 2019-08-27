### Speaker Identification using raw audio with constraint based Convolution Neural Network 

<p> This project fulfilled the requirement of our major project for the final year in Bachelor of Engineering. This is a deep learning project which identifies speaker based on audio they provide. There are several methods to achieve the objective such as MFCC feature extraction, GMM-HMM modeling, CNN, CNN-LSTM and others. Here, inspired by Mirco Ravanelli on his work of raw audio processing using SincNet, we followed his footsteps to create speaker identification system for Nepali enduser. For further understanding of his work, visit: Mirco Ravanelli, Yoshua Bengio, "Speaker Recognition from raw waveform with SincNet" !(https://arxiv.org/abs/1808.00158).</p><hr/>

#### Application
With advent of internet and communication, increasing number of people and applications have established a market over network connecting nodes of mobile and servers across the globe. With information sharing and communicating over this virtual world, a sense of protection is always required. For general people, it comes to the question of privacy and safety of oneself from possible future violation and ransom. For business, it incorporates transactions worth of thousand of millions per day, preventing business from going down to the wrong hands and thus losing customers. Also, it involves data confidentiality and infrastructure security to prevent evasdropping and ultimately saving the company from being destroyed in intellectual level. For governmental level, internet security occupies a significant position to prevent terrorism, war, secret affairs and military advancement.

Using passcode, digital cards and bar code, human authorization have been becoming a matter of lesser importance when to comes to the comparision with biometrics. Biometrics are such entities present with human at all times, mostly. Fingerprint, facial recognition, retina scan are few technology implemented so far and have been way better than traditional password system. Similarly, speech technology have taken their place in biometric technology since mid 20th century. Speaker Identification is one of them which can be tex dependent or text independent based on scope of usage. The features that can be used to remember and distinguish between persons can be formants, accent, emphasize on words, facial expression/gesture, facial structure, speed of speech and so on. All these factors to be incorporated in a system requires a heavy lifting in finding such data, computing platform and of course optimum algorithm to model them together. 

Speaker Identification can be applicable in business platform for online booking and ticketing for cinema, hotel, cab, airplane, money transfer and other. Also, it has application in cloud for data access and platform usage with data encryption and transmission control. In extension to speech technology, it can be used as automated portal of voice command for remote access. In field of emergency call, it can be integrated for automatic call or sms system.   
<hr/>

#### Network Architecture
<p> Network architecture consists of constrained CNN at input that learns features from preprocessed time series of audio. Sinc implementation has been implemented in constrained CNN that extracts cut off frequencies from the input signal. Cut off frequencies are unique and important key factors that distinguish between people. Standard CNN has been used after first CNN to assist in learning while training. Dense layers follow afterwards that adjust network to remember features learnt. 
<p> ![Network Architecture](http://url/to/img.png) </p>
In order to tune parameters other layers such as MaxPooling, Leaky ReLU, Layer Normalization, Batch Normalization, Dropout and L2 regularizers were also used after CNN and Dense layers.</p>
<hr/>
 
#### Environment Setup

<h3> 1. Install Python </h3>
<ul>
  <li> https://realpython.com/installing-python </li>
</ul>

<h3> 2. Enable environment for project installation via virtualenv or conda </h3>
<ul>
  <li> https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ </li>
  <li> https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html </li>
</ul>

<h3> 3. Install following python modules </h3>
<ul>
  <li> tensorflow-gpu </li>
  <li> keras </li>
  <li> flask </li>
  <li> tqdm </li> 
  <li> numpy==1.16.2 </li>
  <li> scipy </li>
  <li> h5py </li>
  <li> pyQT </li>
  <li> librosa </li>
  <li> ffmpeg </li>
<p> list them in separate file titled requirments.txt and install it using pip </p>
<h6> user/path/> pip install requirements.txt </h6>
</ul>

<h3> 4. Database MySQL Xampp </h3>
<p> Start Wampp or Xampp or Lampp web server. Create Database as per requirment.</p>
<p> Our implementation of user information was limited to ID, Name, PhotoPath and  Academic RollNumber. </p>

<h3> 5. Flask: <i> API request for identification </i></h3>
<ul>
  <p> Create virtualenv using cmd:
  <li><h6> user/path/> python -m venv <venv_name></h6></li>
  <li><h6> user/path/> source flaskapi/bin/activate </h6></li>
  <p> Set environment for flask as:
  <li><h6> user/path/> cd app </h6></li>
  <li><h6> user/path/app> set FLASK_APP = api </h6></li>
  <li><h6> user/path/app> set FLASK_ENVIRONMENT = development </h6></li>
  <li><h6> user/path/app> run flask </h6></li>
</ul>

<h3> 4. GUI </h3>
<ul>
  <li> <h6> user/path/> cd GUI </h6></li>
  <li> <h6> user/path/GUI> python main.py </h6></li>
</ul>
