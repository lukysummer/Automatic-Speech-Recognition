from keras.models import Model
from keras.layers import BatchNormalization, Conv1D, Dense, Input, Dropout, TimeDistributed, Activation, Bidirectional, LSTM


def ASR_network(n_input_channels,
                # CNN parameters
                n_cnn_filters,
                kernel_size, 
                stride, 
                padding_mode,
                cnn_dropout,
                dilation,
                # RNN parameters
                n_bdrnn_layers,
                n_hidden_rnn,
                input_dropout,      
                recurrent_dropout,  
                rnn_merge_mode,
                # FC (Fully Connected) parameters
                fc_n_hiddens,
                fc_dropout,
                output_dim = 29):

    # 1. Main acoustic input
    input_data = Input(name = 'input', 
                       shape = (None, n_input_channels))
    
    
    # 2. Add a convolutional layer & dropout
    out = Conv1D(filters = n_cnn_filters, 
                 kernel_size = kernel_size, 
                 strides = stride, 
                 padding = padding_mode,
                 activation = 'relu',
                 dilation_rate = dilation,
                 name = 'cnn')(input_data)
    
    
    # 3. Add dropout & batch normalization for cnn output
    out = Dropout(cnn_dropout, name='dropout_cnn')(out)
    out = BatchNormalization(name = 'bn_cnn')(out)
  
    
    # 4. Add bi-directional recurrent layers & batch normalizations
    for i in range(n_bdrnn_layers):
        out = Bidirectional(LSTM(units = n_hidden_rnn,
                                 activation = 'tanh',
                                 return_sequences = True,
                                 dropout = input_dropout,
                                 recurrent_dropout = recurrent_dropout,
                                 implementation = 2), 
                            merge_mode = rnn_merge_mode,
                            name = 'bdrnn_'+ str(i+1))(out)

        out = BatchNormalization(name = 'bn_rnn_' + str(i+1))(out)
    
    
    # 5. Add a TimeDistributed Dense layers
    for i, n_hidden in enumerate(fc_n_hiddens + [output_dim]):
        # if not the last FC layer, perform dropout & relu activation
        if i < len(fc_n_hiddens):
            out = TimeDistributed(Dense(n_hidden), name = "td_dense_" + str(i+1))(out)
            out = Dropout(fc_dropout, name='dropout_fc_' + str(i+1))(out)
            out = Activation('relu', name = 'fc_relu_' + str(i+1))(out)
            
        # if the last FC layer, perform softmax activation
        else:
            final_fc_out = TimeDistributed(Dense(n_hidden), name = "td_dense_" + str(i+1))(out)
            y_pred = Activation('softmax', name = 'fc_softmax')(final_fc_out)

    
    # 8. Build the model
    model = Model(inputs = input_data, 
                  outputs = y_pred)
    
    
    # 9. Define output_length function
    model.output_length = lambda x: cnn_output_length(x, 
                                                      kernel_size, 
                                                      padding_mode, 
                                                      stride)
    
    print(model.summary(line_length=110))
    
    return model



def cnn_output_length(input_length, 
                      filter_size, 
                      border_mode, 
                      stride,
                      dilation = 1):
    """ 
    :Computes the length of the output sequence after 1D convolution along time
    Params:
        input_length (int): Length of input sequence
        filter_size (int): Kernel size of 1D convultion
        border_mode (str): CNN padding mode; only supports `same` or `valid`
        stride (int): Stride of  1D convolution
        dilation (int): Length of dilation
    """
    if input_length is None:
        return None
    
    assert border_mode in {'same', 'valid'}
    
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    
    if border_mode == 'same':
        output_length = input_length
    
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1

    return (output_length + stride - 1) // stride