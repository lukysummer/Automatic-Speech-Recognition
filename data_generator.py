import json
import numpy as np
from sepctrogram_generator import spectrogram_from_file
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from char_map import char_map, index_map
from char_map import text_to_int


class AudioGenerator():
    
    def __init__(self,
                 max_length = 10.0,
                 max_freq = 8000,
                 spectrogram = True,
                 frame_window = 20,
                 frame_stride = 10,
                 mfcc_dim = 16,
                 batch_size = 20):
        
        self.max_length = max_length
        self.max_freq = max_freq
        
        # Spectrogram params:
        self.spectrogram = spectrogram
        self.spectrogram_n_frames = int(0.001 *frame_ window * max_freq) + 1  # in ms
        self.frame_window = frame_window
        self.frame_stride = frame_stride
        
        self.mfcc_dim = mfcc_dim
        self.batch_size = batch_size
        
        self.current_train_idx = 0
        self.current_valid_idx = 0
            
    
    def load_audio(self, filepath,
                         partition):
    
        filepaths, lengths, texts = [], [], []
        
        with open(filepath) as json_line_file:
            for line_i, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line) # dict
                    if float(spec['duration']) > self.max_length:
                        continue
                    filepaths.append(spec['key'])
                    lengths.append(spec['duration'])
                    texts.append(spec['text'])
                    
                except Exception as e:
                    print("Error reading line #{}: {}".format(line_i, json_line))
        
        if partition == 'train':
            self.train_filepaths = filepaths
            self.train_lengths = lengths
            self.train_texts = texts
            
        if partition == 'valid':
            self.valid_filepaths = filepaths
            self.valid_lengths = lengths
            self.valid_texts = texts
            
    
    ### Transform the signal input into a 2D SPECTROGRAM or a 2D MFCC FEATURE ###
    def featurize(self, filepath):
        if self.spectrogram:
            return spectrogram_from_file(filepath,
                                         frame_stride = self.frame_stride,
                                         frame_window = self.frame_window,
                                         max_freq = self.max_freq) 
            
        else:
            (sample_rate, signal) = wav.read(filepath)
            
            return mfcc(signal, sample_rate, numcep = self.mfcc_dim)
            
        
    def fit_train(self, k_samples = 100):
        '''
        Estimate mean & std of training set features (to normalize data)
        : param k_samples (int): number of samples to use for estimation
        '''
        k_samples = min(k_samples, len(self.train_filepaths))
        audio_samples = np.random.choice(self.train_filepaths, k_samples)
        features = [self.featurize(audio) for audio in audio_samples]
        stacked_features = np.vstack(features)  
        # [sum of n_frames for all k_samples spectrograms, # of frequency bins]
        self.feature_mean = np.mean(stacked_features, axis = 0)  #[# of frequency bins,]
        self.feature_std = np.std(stacked_features, axis = 0) #[# of frequency bins,]
    
    
    def load_train_data(self, filepath):
        self.load_data(filepath, partition = 'train')
        self.fit_train()

    
    def load_valid_data(self, filepath):
        self.load_data(filepath, partition = 'valid')
    
    
    def normalize(self, feature, eps = 1e-14):
        return (feature - self.feature_mean) / (self.feature_std + eps)
        
    
    def get_batch(self, partition):
        
        if partition == "train":
            filepaths = self.train_filepaths
            current_idx = self.current_train_idx
            texts = self.train_texts
            
        if partition == "valid":
            filepaths = self.valid_filepaths
            current_idx = self.current_valid_idx
            texts = self.valid_texts
        
        features = [self.normalize(self.featurize(audio)) for audio in \
                    filepaths[current_idx : current_idx + self.batch_size]]
        
        # longest audio's length:
        max_audio_len = max([features[i].shape[0] for i in range(self.batch_size)])
        max_text_len = max([len(texts[current_idx + i] for i in range(self.batch_size))])
        
        if self.spectrogram:
            X = np.zeros([self.batch_size, max_audio_len, self.spectrogram_n_frames])
        else:
            X = np.zeros([self.batch_size, max_audio_len, self.mfcc_dim])
            
        Y = np.ones([self.batch_size, max_text_len]) * 28
            
        input_lengths = np.zeros([self.batch_size, 1])
        label_lengths = np.zeros([self.batch_size, 1])
        
        for i in range(self.batch_size):
            feat = features[i]
            input_lengths[i] = feat.shape[0]
            X[i, :feat.shape[0], :] = feat
            
            label = np.array(text_to_int(texts[current_idx+i]))
            Y[i, :len(label)] = label
            label_lengths[i] = len(label)
            
        outputs = {'ctc' : np.zeros([self.batch_size])}
        inputs = {'inputs' : X,
                  'labels' : Y,
                  'input_length' : input_lengths,
                  'label_length' : label_lengths}
        
        return (inputs, outputs)
    
    
    def shuffle(self, partition):
        
        if partition == 'train':
            filepaths = self.train_filepaths
            audio_lengths = self.train_lengths
            texts = self.train_texts
        
        elif partition == 'valid':
            filepaths = self.valid_filepaths
            audio_lengths = self.valid_lengths
            texts = self.valid_texts    
            
        p = np.random.permutation(len(filepaths))
        
        self.train_filepaths = [filepaths[i] for i in p]
        self.train_lengths = [audio_lengths[i] for i in p]
        self.train_texts = [texts[i] for i in p]
        
    
    def get_train_batch(self):
        '''
        Obtain a batch of training data
        '''
        while True:
            batch = self.get_batch('train')
            self.current_train_idx += self.batch_size
            
            if self.current_train_idx >= len(self.train_texts) - self.batch_size:
                self.current_train_idx = 0
                self.shuffle('train')
            
            yield batch
            
            
    def get_valid_batch(self):
        '''
        Obtain a batch of validation data
        '''
        while True:
            batch = self.get_batch('valid')
            self.current_valid_idx += self.batch_size
            
            if self.current_valid_idx >= len(self.valid_texts) - self.batch_size:
                self.current_valid_idx = 0
                self.shuffle('valid')
            
            yield batch
     