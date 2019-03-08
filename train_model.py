import os

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   

from data_generator import AudioGenerator
from ASR_model import ASR_network



def ctc_lambda_func(args):
    ''' Returns tensorflow's ctc_batch_cost with given arguments '''
    
    y_pred, labels, input_length, label_length = args
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



def add_ctc_loss(model): 
    ''' Returns model with CTC Loss added  '''
    
    # 1. Define Inputs
    labels = Input(name = 'labels', 
                   shape = (None,), 
                   dtype = 'float32')
    
    input_lengths = Input(name = 'input_length', 
                          shape = (1,), 
                          dtype = 'int64')
    
    label_lengths = Input(name = 'label_length', 
                          shape=(1,),  
                          dtype='int64')
    
    # 2. Define Lambda function for output lengths
    output_lengths = Lambda(model.output_length)(input_lengths)
    
    # 3. Implement CTC loss via Lambda layer
    loss_out = Lambda(ctc_lambda_func, 
                      output_shape = (1,), 
                      name = 'ctc')([model.output, labels, output_lengths, label_lengths])
    
    # 4. Build the model with CTC loss
    model = Model(inputs = [model.input, labels, input_lengths, label_lengths], 
                  outputs = loss_out)
    
    return model



def train_model(model, 
                n_epochs,
                lr,
                save_model_path,
                spectrogram = True,
                train_json = 'train_corpus.json',
                valid_json = 'valid_corpus.json',
                minibatch_size = 20,               
                mfcc_dim = 13,
                verbose = 1,
                max_duration = 10.0):
    ''' Trains the model and saves the weights '''
    
    # 1. Create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size = minibatch_size, 
                               spectrogram = spectrogram, 
                               mfcc_dim = mfcc_dim, 
                               max_duration = max_duration)
    
    # 2. Add Training & Validation Data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)
    
    # 3. Calculate steps_per_epoch
    num_train_examples = len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size
    
    # 4. Calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths) 
    validation_steps = num_valid_samples//minibatch_size
    
    # 5. Add CTC loss to the NN specified in model
    model_with_ctc = add_ctc_loss(model)
    
    # 6. Compile the model with training parameters
    # *CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model_with_ctc.compile(loss = {'ctc': lambda y_true, y_pred: y_pred}, 
                           optimizer = SGD(lr = lr, 
                                           decay = 1e-6, 
                                           momentum = 0.9, 
                                           nesterov = True, 
                                           clipnorm = 5.))

    # 7. Make "results/" directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # 8. Add checkpointer
    checkpointer = ModelCheckpoint(filepath='results/'+save_model_path, verbose=0)

    # 9. Train the model
    hist = model_with_ctc.fit_generator(generator = audio_gen.next_train(), 
                                        steps_per_epoch = steps_per_epoch,
                                        epochs = n_epochs, 
                                        validation_data = audio_gen.next_valid(), 
                                        validation_steps = validation_steps,
                                        callbacks = [checkpointer], 
                                        verbose = verbose)

        
############################## TRAIN THE MODEL ################################
n_epochs = 43
lr = 0.02
save_model_path = "model_epoch_" + n_epochs + ".h5"
use_spectrogram = True

train_model(model = ASR_network, 
            n_epochs = n_epochs,
            lr = lr,
            save_model_path = 'model_final_43_epochs.h5',
            spectrogram = use_spectrogram)