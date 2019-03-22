import os
import _pickle as pickle

from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from Data_Generator import AudioGenerator
from ASR_model import ASR_network


def get_ctc_loss(args):
    ''' Compute CTC loss with given arguments '''
    
    model_outputs, model_output_lengths, true_labels, true_label_lengths  = args
    
    return K.ctc_batch_cost(y_true = true_labels, 
                            y_pred = model_outputs, 
                            input_length = model_output_lengths, 
                            label_length = true_label_lengths) 



def add_ctc_loss(model_without_ctc):
    ''' Returns the model with CTC loss added as Output '''
    
    ############## 1. Define Additional Inputs for the New Model ##############
    true_labels = Input(shape = (None, ), name = "labels", dtype = "float32")
    input_lengths = Input(shape = (1,), name = "input_lengths", dtype = "int64")
    label_lengths = Input(shape = (1,), name = "label_lengths", dtype = "int64")
    
    
    ################## 2. Define Lengths of Model Outputs  ####################
    model_output_lengths = Lambda(model_without_ctc.output_length)(input_lengths)
    
    
    ######################## 3. Define CTC Loss Layer #########################
    ''' 
    Since Keras doesn NOT support loss function with more than 1 parameters, 
    implement CTC loss in a Lambda layer 
    '''
    ctc_loss = Lambda(function = get_ctc_loss,
                      output_shape = (1,),
                      name = 'ctc')([model_without_ctc.output,
                                     model_output_lengths,
                                     true_labels,  
                                     label_lengths])
    
    
    ######################## 4. Define the New Model ##########################
    model_with_ctc = Model(inputs = [model_without_ctc.input, 
                                    true_labels, 
                                    input_lengths, 
                                    label_lengths],
                           outputs = ctc_loss)
    
    return model_with_ctc
    


def train_model(ASR_model_without_ctc,
                n_epochs,
                lr,
                save_loss_path,
                save_model_path,
                train_corpus_path,
                valid_corpus_path,
                verbose = 1):
    
    ################### 1. Create Audio Generator Instance ####################
    batch_size = 20
    n_epochs = n_epochs
    
    audio_generator = AudioGenerator(batch_size = batch_size)
    audio_generator.load_train_data(train_corpus_path)
    audio_generator.load_valid_data(valid_corpus_path)
    
    train_steps = audio_generator.num_train_audios//batch_size
    valid_steps = audio_generator.num_valid_audios//batch_size
    
    
    ################### 2. Add CTC loss layer to the model ####################
    ASR_model_with_ctc = add_ctc_loss(ASR_model_without_ctc)
    ''' 
    - Since CTC loss is implemented as the OUTPUT of the model (= y_pred), 
      use a dummy Lambda function for the loss: lambda y_pred, y_true: y_pred.
    - y_true was defined as 0 in AudioGenerator
    *Note: All standard loss functions in Keras uses a formula containing y_pred & y_true 
    '''
    ASR_model_with_ctc.compile(loss = {'ctc': lambda y_true, y_pred: y_pred},
                               optimizer = SGD(lr = lr, 
                                               decay = 1e-6,
                                               momentum = 0.9, 
                                               nesterov = True,
                                               clipnorm = 5.))
    
    
    ############## 3. Add Checkpointer to Save Model Parameters ###############
    if not os.path.exists('results'):
        os.makedirs('results')
        
    checkpointer = ModelCheckpoint(filepath = 'results/' + save_model_path, verbose = 0)
    
    
    ########################## 4. Train the Model #############################
    loss_history = ASR_model_with_ctc.fit_generator(generator = audio_generator.next_train_batch(),
                                                    steps_per_epoch = train_steps, 
                                                    epochs = n_epochs, 
                                                    validation_data = audio_generator.next_valid_batch(), 
                                                    validation_steps = valid_steps, 
                                                    callbacks = [checkpointer],
                                                    verbose = verbose)
    
    
    ########################## 5. Save Model Losses ###########################
    with open('results/' + save_loss_path, 'wb') as f:
        pickle.dump(loss_history.history, f)
        
        

ASR_net = ASR_network(n_input_channels = 161,
                     # cnn params
                     n_cnn_filters = 200,
                     kernel_size = 11, 
                     stride = 2, 
                     padding_mode = 'valid',
                     dilation = 1,
                     cnn_dropout = 0.3,
                     # rnn params
                     n_bdrnn_layers = 2,
                     n_hidden_rnn = 200,
                     input_dropout = 0.3,      # dropout values referenced from: 
                     recurrent_dropout = 0.1,  # https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/
                     rnn_merge_mode = 'sum',
                     # fc params
                     fc_n_hiddens = [200],
                     fc_dropout = 0.3,
                     output_dim = 29)       


n_epochs = 30
lr = 0.2

train_model(ASR_net,
            n_epochs = n_epochs,
            lr = lr,
            save_loss_path = "losses_" + n_epochs + "_epochs.pickle",
            save_model_path = "model_" + n_epochs + "_epochs.h5",
            train_corpus_path = "data/train_small_corpus.json",
            valid_corpus_path = "data/valid_small_corpus.json",
            verbose = 1)