import json
import numpy as np
from spectrogram_generator import spectrogram_from_file
from char_map import text_to_int


class AudioGenerator():
    
    def __init__(self, 
                 max_duration = 10.0, # (for audios) in seconds
                 max_freq = 8000,     # maximum threshold for spectrogram frequency
                 batch_size = 20):
        
        self.max_duration = max_duration
        self.max_freq = max_freq
        
        self.batch_size = batch_size
        
        self.train_index = 0
        self.valid_index = 0
        
        
    
    ############################## 1. LOAD DATA ###############################
    def load_data(self, json_path, partition):
        
        input_paths = []
        spectrograms = []
        durations = []
        labels = []       

        with open(json_path) as json_file:
            for line in json_file:
                metadata = json.loads(line)
                if metadata["duration"] > self.max_duration:
                    continue
                input_paths.append(metadata["key"])
                spectrograms.append(spectrogram_from_file(metadata["key"], max_freq = self.max_freq))
                durations.append(metadata["duration"])
                labels.append(metadata["text"])
                
                
        if partition == "train":
            self.num_train_audios = len(input_paths)
            self.train_spectrograms = spectrograms
            self.train_labels = labels
            
            # Find mean & standard deviation along temporal dimension for random 500 spectrograms
            random_500 = np.random.choice(self.train_spectrograms, 500, replace=False)
            stacked_500 = np.vstack(random_500)
            self.train_mean = np.mean(stacked_500, axis = 0)
            self.train_std = np.std(stacked_500, axis = 0) + 1e-14            
            
        elif partition == "valid":
            self.num_valid_audios = len(input_paths)
            self.valid_spectrograms = spectrograms
            self.valid_labels = labels
    

    def load_train_data(self, json_path):
        self.load_data(json_path = json_path, partition = "train")
        
        
    def load_valid_data(self, json_path):
        self.load_data(json_path = json_path, partition = "valid")
        
    
    
    ############################## 2. BATCH DATA ##############################    
    def next_batch(self, partition):
        ################# (2.1) Assign Data by Partition #################
        if partition == "train":
            index = self.train_index
            spectrograms = self.train_spectrograms
            labels = self.train_labels
                       
        elif partition == "valid":
            index = self.valid_index
            spectrograms = self.valid_spectrograms
            labels = self.valid_labels
        
        
        #################### (2.2) Initialize Input Arrays ####################
        # Contruct array of maximum temporal/label length for the CURRENT BATCH
        max_audio_length = max([sp.shape[0] for sp in spectrograms[index:index + self.batch_size]])
        max_text_length = max([len(label) for label in labels[index:index + self.batch_size]])
        
        batch_inputs = np.zeros([self.batch_size, max_audio_length, spectrograms[0].shape[1]])  # spectrograms
        batch_labels = np.ones([self.batch_size, max_text_length])  # true text labels
        input_lengths = np.zeros([self.batch_size, 1])
        label_lengths = np.zeros([self.batch_size, 1])
        
        
        ################# (2.3) Assign Values to Input Arrays #################
        for i in range(self.batch_size):
            
            spectrogram = spectrograms[index + i]
            encoded_text = text_to_int(labels[index + i])
            
            # Normalize along temporal dimension
            spectrogram = (spectrogram - self.train_mean)/self.train_std
            
            batch_inputs[i, :spectrogram.shape[0], :] = spectrogram
            batch_labels[i][:len(encoded_text)] = encoded_text
            
            input_lengths[i] = spectrogram.shape[0]
            label_lengths[i] = len(encoded_text)


        ####################### (2.4) Update batch index ######################
        if partition == "train":
            self.train_index += self.batch_size
            # if already went over the whole training data, reset batch index = 0 & shuffle data
            if self.train_index > self.num_train_audios - self.batch_size:
                self.train_index = 0     
                permutated_idx = np.random.permutation(range(self.num_train_audios))
                # shuffle training data:
                self.train_spectrograms = [self.train_spectrograms[idx] for idx in permutated_idx]
                self.train_labels = [self.train_labels[idx] for idx in permutated_idx]

        elif partition == "valid":
            self.valid_index += self.batch_size
            # if already went over the whole validation data, reset batch index = 0 & shuffle data
            if self.valid_index > self.num_valid_audios - self.batch_size:
                self.valid_index = 0
                permutated_idx = np.random.permutation(range(self.num_valid_audios))
                # shuffle validation data:
                self.valid_spectrograms = [self.valid_spectrograms[idx] for idx in permutated_idx]
                self.valid_labels = [self.valid_labels[idx] for idx in permutated_idx]        
        
        
        ################# (2.5) Construct Input & Output Dicts ################        
        inputs = {"inputs" : batch_inputs,          # spectrograms
                  "labels" : batch_labels,         # encoded true text
                  "input_lengths" : input_lengths,  # required for CTC loss
                  "label_lengths" : label_lengths}  # required for CTC loss
        # CTC loss to be defined later as the output of the model
        output = {"ctc": np.zeros(self.batch_size)}       
                
        return (inputs, output)
 

    def next_train_batch(self):
        while True:
            yield self.next_batch("train")
        
        
    def next_valid_batch(self):
        while True:
            yield self.next_batch("valid")