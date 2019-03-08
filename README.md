# Automatic Speech Recognition in Keras

<p align="center"><img src="images/pipeline.png" height = "256"></p>

This is my implementation of Automatic Speech Recognition using Convolutional and Recurrent Neural Networks in Keras, 
as a project for Udacity's Natural Language Processing Nanodegree ([Course Page](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892)).



## Results

Following are some of the example outputs of the model (PRED) along with the true transcription (TRUE):


* (TRUE) "her father is a most remarkable person to say the least"  

--> (PRED) **"her father s a most ere markcabl person to say the last"**
* (TRUE) "he gave thanks for our food and comfort and prayed for the poor" 

--> (PRED) **"he gave think s for a foodant comfort and pride for the poor"**



## Repository 

This repository contains:
* **data/** folder contains: 

Train and Validation JSON files contain lines in a dictionary format: {"key": path to the .wav audio file, "duration": length of the audio file in senconds, "text": true transcription of the audio in text}
/t * **"train_small_corpus.json"**: Training Corpus of 2703 sentences
/t * **"valid_small_corpus.json"**: Validation Corpus of 2620 sentences
* **spectrogram_generator.py** : Code for generating spectrograms from raw audio files [Source](https://github.com/baidu-research/ba-dls-deepspeech)
* **char_map.py** : Helper function for encoding each letter into int 
* **data_generator.py** : Complete code for building, training, and making inference from seq2seq model in Keras
* **ASR_model.py** : Complete Deep Neural Network Model (in Keras) for ASR
* **train_model.py** : Code for training the ASR model with CTC loss
* **predict_text.py** : 
* **ASR_Step_by_Step_Notebook.ipynb** : step-by-step Jupyter Notebook for for building, training, and making inference from the ASR model



## List of Hyperparameters Used:
INPUT:


* Input Feautres = Spectrogram (w/ varying temporal length & 161 Frequency levels)


* 1D CNN LAYERS ALONG TEMPORAL DIMENSION:


* Number of Layers: **1**
* Number of Filters: **200**
* Kernel Size: **11**
* Stride: **2**
* Activation: **ReLU**
* Padding Mode: **'valid'**
* Dilation: **1**
* Dropout: **30%**
* Order: **Activation -> Dropout -> Batch Normalization**


RNN LAYERS:


* Type: **Bidirectional LSTM**
* Number of Layers : **2**
* Number of Hidden Nodes: **200**
* Activation: **Tanh**
* Input Dropout: **30%**
* Recurrent Layer Dropout: **10%**
* RNN Merge Mode: **'sum'**


FULLY CONNECTED LAYERS:


* Number of Layers: **2**
* Number of Hidden Nodes: **200 (1st), 29 (2nd)**
* Activation: **ReLU (1st), Softmax (2nd)**
* Dropout: **30%**


TRAINING:


* Loss: **CTC Loss***
* Batch Size: **20**
* Number of Epochs: **43**
* Learning Rate = **0.02 (first 30 epochs), 0.01 (epochs 31-40), 0.003 (last 3 epochs)**



## Sources


* Paper: ["Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks"](https://arxiv.org/pdf/1701.02720.pdf)
* Udacity's Natural Language Processing Nanodegree's workspace ([Course Page](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892))
