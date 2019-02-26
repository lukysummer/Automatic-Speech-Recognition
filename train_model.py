from data_generator import AudioGenerator

from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import os

################# 1. LOAD AUDIO (input) & TEXT (label) DATA ###################
def get_CTC_loss(preds, labels, input_lengths, label_lengths):
    
    return K.ctc_batch_cost(labels, preds, input_lengths, label_lengths)


def get_model_w_CTC_loss(model):
    
    labels = Input(name = "labels", 
                   shape = (None,),
                   dtype = "float32")
    
    label_lengths = Input(name = "label_lengths",
                          shape = (1,),
                          dtype = "int64")
    
    input_lengths = Input(name = "input_lengths",
                          shape = (1,),
                          dtype = "int64")
    
    output_lengths = Lambda(model.output_length)(input_lengths)

    ctc_loss = Lambda(get_CTC_loss,
                      output_shape = (1,),
                      name = 'ctc_loss')[model.output,
                                         labels,
                                         output_lengths,
                                         label_lengths]
    
    model_w_CTC_loss = Model(inputs = [model.input,
                                       labels,
                                       input_lengths,
                                       label_lengths],
                             output = ctc_loss)
    
    return model_w_CTC_loss


batch_size = 20

audio_generator = AudioGenerator(batch_size = batch_size,
                                 spectrogram = True,
                                 mfcc_dim = 16,
                                 max_length = 10.0)

audio_generator.load_train_data(filepath = "train_corpus.json")
audio_generator.load_valid_data(filepath = "train_valid.json")

n_batches = len(audio_generator.train_filepaths) // batch_size
n_valid_batches = len(audio_generator.valid_filepaths) // batch_size

model_w_CTC_loss = get_model_w_CTC_loss(model)


# CTC loss is implemented separately, so use a dummy lambda function for the loss:
model_w_CTC_loss.compile(loss = {'ctc': lambda y_true, y_pred: y_pred},
                         optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5))


###### 8. Add checkpointer #####
if not os.path.exists('results'):
    os.makedirs('results')
        
checkpointer = ModelCheckpoint(filepath = 'results/model.h5', 
                               verbose = 1)

############################# 9. TRAIN THE NETWORK ############################
n_epochs = 20

history = model_w_CTC_loss.fit_generator(generator = audio_generator.get_train_batch(),
                                         stps_per_epoch = n_batches,
                                         epochs = n_epochs,
                                         validation_data = audio_generator.get_valid_batch(),
                                         validation_steps = n_valid_batches,
                                         callbacks = [checkpointer],
                                         verbose = 1)
