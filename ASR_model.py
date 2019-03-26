from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, GRU, TimeDistributed, Dense
from keras.layers import Input, Permute, BatchNormalization, Dropout
from keras.models import Model


def ASR_network(spectrogram_dim = 161,
                # CNN params
                n_layers_cnn = 2,
                n_hidden_cnn = [128, 256],
                kernel_size = 11,
                stride = 2,
                dilation = 2,
                padding_mode_cnn = "valid",
                activation_cnn = "relu",
                dropout_cnn = 0.3,
                # Max Pooling params
                pooling_size = 4,
                pooling_stride = 2, 
                # RNN params
                type_rnn = "LSTM",
                n_layers_rnn = 2,
                n_hidden_rnn = 200,
                activation_rnn = "tanh",
                merge_mode_rnn = "sum",
                dropout_rnn_input = 0.3,
                dropout_rnn_recur = 0.1,
                # FC params
                n_layers_fc = 2,
                n_hidden_fc = [200, 29],
                dropout_fc = 0.3,
                activation_fc = "relu"):
    
    assert type_rnn in {"LSTM", "GRU"}, "RNN must be either LSTM or GRU."
    
    ########################## 1. Spectrogram Inputs ##########################
    spectrograms = Input(name = "inputs", 
                         shape = (None, spectrogram_dim))
    
    
    ########### 2. 1D Convolutional Layers Over Temporal Dimension ############
    for i in range(n_layers_cnn):       
        
        CNN = Conv1D(name = "1D_CNN_" + str(i+1),
                     filters = n_hidden_cnn[i],
                     kernel_size = kernel_size,
                     strides = stride,
                     dilation_rate = dilation,
                     padding = padding_mode_cnn,
                     activation = activation_cnn)
        
        if i == 0:
            batch_x = CNN(spectrograms)
            
            # Max Pooling across Frequency dimension for 1st Layer ONLY
            batch_x = Permute((2, 1))(batch_x)
            
            padding_mode_maxpool = padding_mode_cnn if padding_mode_cnn != "causal" else "same"
            batch_x = MaxPooling1D(name = "Max_Pooling",
                                   pool_size = pooling_size,
                                   strides = pooling_stride,
                                   padding = padding_mode_maxpool)(batch_x)
            
            batch_x = Permute((2, 1))(batch_x)
            
        else:
            batch_x = CNN(batch_x)
        
        
        
        batch_x = Dropout(name = "Dropout_CNN_" + str(i+1), 
                          rate = dropout_cnn,)(batch_x)

        batch_x = BatchNormalization(name = "BatchNorm_CNN_" + str(i+1))(batch_x)
        
        
    ###################### 3. Bi-directional RNN Layers #######################
    for i in range(n_layers_rnn):

        if type_rnn == "LSTM":
            batch_x = Bidirectional(LSTM(units = n_hidden_rnn,
                                         activation = activation_rnn,
                                         return_sequences = True,
                                         implementation = 2,
                                         dropout = dropout_rnn_input,
                                         recurrent_dropout = dropout_rnn_recur),
                                    merge_mode = merge_mode_rnn,
                                    name = "BD_LSTM_" + str(i+1))(batch_x)
            
        elif type_rnn == "GRU":
            batch_x = Bidirectional(GRU(units = n_hidden_rnn,
                                        activation = activation_rnn,
                                        return_sequences = True,
                                        implementation = 2,
                                        dropout = dropout_rnn_input,
                                        recurrent_dropout = dropout_rnn_recur),
                                    merge_mode = merge_mode_rnn,
                                    name = "BD_GRU_" + str(i+1))(batch_x)

        batch_x = BatchNormalization(name = "BatchNorm_RNN_" + str(i+1))(batch_x)
        
        
    ####################### 4. Fully Connected Layers #########################
    for i in range(n_layers_fc):
        # if not last fc layer
        if i < (n_layers_fc - 1):
            batch_x = TimeDistributed(Dense(units = n_hidden_fc[i], 
                                            activation = activation_fc), name = "FC_" + str(i+1))(batch_x)
                
            batch_x = Dropout(name = "Dropout_FC_" + str(i+1),
                              rate = dropout_fc)(batch_x)
                
        # Final Prediction Layer: softmax activation    
        else:
            logits = TimeDistributed(Dense(units = n_hidden_fc[i],
                                            activation = "softmax"), name = "Final_FC")(batch_x)
    

    ############################# 5. Final Model ##############################        
    asr_model = Model(inputs = spectrograms,
                      outputs = logits)
    
    
    ################ 6. Output (Temporal) Length (for CTC Loss) ###############
    asr_model.output_length = lambda x : cnn_output_length(x,
                                                           n_layers_cnn,
                                                           kernel_size,
                                                           stride,
                                                           padding_mode_cnn,
                                                           dilation,
                                                           pooling_size,
                                                           pooling_stride)
    
    print(asr_model.summary(line_length=110))
    
    return asr_model

            

def cnn_output_length(length,  # Input Length
                      n_layers_cnn,
                      kernel_size,
                      stride,
                      padding_mode_cnn,
                      dilation,
                      pooling_size,
                      pooling_stride):
    """ Computes the output length (in temporal dimension) after 1D convolution along time """
    
    assert padding_mode_cnn in {"valid", "same", "causal"}, "CNN's Padding mode must be one of 'valid', 'same', or 'causal'."
    
    if length is None:
        return None
    
    if padding_mode_cnn in ["same", "causal"] :
        for i in range(n_layers_cnn):
            length = (length//stride) + int(not(length % stride == 0))   
            # for maxpooling
            if i == 0:
                length = (length//pooling_stride) + int(not(length % pooling_stride == 0))   
    
    elif padding_mode_cnn == "valid":  # no padding
        for i in range(n_layers_cnn):
            length = length - kernel_size*dilation + dilation - 2
            # for maxpooling
            if i == 0:
                length = (length - pooling_size)//pooling_stride + 1
            
    return length