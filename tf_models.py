import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Conv2DTranspose, CuDNNLSTM, Bidirectional, RepeatVector, Concatenate
import tensorflow.keras.backend as K


# Just rnn based multi-layer network
def Basic(feed_list, config):
    x = feed_list[0]
    
    inputs_dims = config.inputs_dims
    n_layers = config.n_layers
    no_bidirectional = config.no_bidirectional
    model_type = config.model_type
    feature_dict = {}
    
    for i in range(n_layers):
        n_dims = inputs_dims if i != n_layers-1 else inputs_dims//2
        layer = CuDNNLSTM(units=n_dims, kernel_initializer='he_normal', return_sequences=True, name=f'layer_{i}')
        if no_bidirectional != True: layer = Bidirectional(layer)

        x = layer(x)
        feature_dict[i] = x
        
    outputs = x
    
    loss_dict = {}
    classifier_loss = None
    recons_loss = tf.reduce_sum(tf.pow(outputs - feed_list[0], 2))
    loss = classifier_loss + recons_loss if classifier_loss is not None else recons_loss
    
    loss_dict['classifier_loss'] = classifier_loss
    loss_dict['recons_loss'] = recons_loss
    loss_dict['loss'] = loss
    
    return outputs, loss_dict, feature_dict
    

# Autoencoder by Seq2Seq
def Autoencoder(feed_list, config):
    x = feed_list[0]
    
    inputs_dims = config.inputs_dims
    timesteps = config.timesteps
    n_layers = config.n_layers
    no_bidirectional = config.no_bidirectional
    model_type = config.model_type
    option = config.option
    feature_dict = {}
    
    """  Encoder  """
    for i in range(n_layers//2):
        n_dims = inputs_dims // (2**(i+2))
        return_sequences = True
        return_state = False if i != (n_layers//2)-1 else True
        layer = CuDNNLSTM(units=n_dims, kernel_initializer='he_normal', return_sequences=return_sequences, return_state=return_state, name=f'layer_{i}')
        if no_bidirectional != True:
            layer = Bidirectional(layer)

        if i != (n_layers//2)-1: 
            x = layer(x)
        else:
            x_outputs = layer(x)
            x = x_outputs[0]
            encoder_state = x_outputs[1:]
        
        feature_dict[i] = x
    
    state_h = tf.concat([encoder_state[0], encoder_state[2]], axis=-1)
    state_c = tf.concat([encoder_state[1], encoder_state[3]], axis=-1)
    
    if option == 'context': # Use last encoder state as `context vector` to decoder input
        context = RepeatVector(timesteps)(state_h) # Decoder inputs
        x = tf.concat([x, context], axis=-1)
        
    feature_dict['laten_sequence'] = x
            
    """ Decoder """
    for j in range(i+1, n_layers):
        n_dims = inputs_dims // (2**(n_layers-j))
        layer = CuDNNLSTM(units=n_dims, kernel_initializer='he_normal', return_sequences=True, name=f'layer_{j}')
        if no_bidirectional != True:
            layer = Bidirectional(layer)
            
        x = layer(x) #if j != i+1 else layer(x, initial_state=[state_h, state_c]) # initial state of first decoder lstm is last encdoer state
        feature_dict[j] = x
        
    outputs = x
        
    loss_dict = {}
    classifier_loss = None
    recons_loss = tf.reduce_sum(tf.pow(outputs - feed_list[0], 2))
    loss = classifier_loss + recons_loss if classifier_loss is not None else recons_loss
    
    loss_dict['classifier_loss'] = classifier_loss
    loss_dict['recons_loss'] = recons_loss
    loss_dict['loss'] = loss
    
    return outputs, loss_dict, feature_dict


# Autoencoder by Seq2Seq
def OneClassAutoencoder(feed_list, config):
    x = feed_list[0]
    labels = feed_list[1]
    
    inputs_dims = config.inputs_dims
    timesteps = config.timesteps
    n_layers = config.n_layers
    no_bidirectional = config.no_bidirectional
    model_type = config.model_type
    option = config.option
    feature_dict = {}
    
    """  Encoder  """
    for i in range(n_layers//2):
        n_dims = inputs_dims // (2**(i+2))
        return_sequences = True
        return_state = False if i != (n_layers//2)-1 else True
        layer = CuDNNLSTM(units=n_dims, kernel_initializer='he_normal', return_sequences=return_sequences, return_state=return_state, name=f'layer_{i}')
        if no_bidirectional != True:
            layer = Bidirectional(layer)

        if i != (n_layers//2)-1: 
            x = layer(x)
        else:
            x_outputs = layer(x)
            x = x_outputs[0]
            encoder_state = x_outputs[1:]

        feature_dict[i] = x
            
    state_h = tf.concat([encoder_state[0], encoder_state[2]], axis=-1)
    score_vector = Dense(units= 2, kernel_initializer='he_normal', name='score_layer')(state_h) # Shape=[batch_size,2]
    if option == 'condition':
        condition = RepeatVector(timesteps)(score_vector)#(score_vector) # Shape=[batch_size, timesteps, 2]
        x = tf.concat([x, condition], axis=-1)
            
    """ Decoder """
    for j in range(i+1, n_layers):
        n_dims = inputs_dims // (2**(n_layers-j))
        layer = CuDNNLSTM(units=n_dims, kernel_initializer='he_normal', return_sequences=True, name=f'layer_{j}')
        if no_bidirectional != True:
            layer = Bidirectional(layer)
            
        x = layer(x) #if j != i+1 else layer(x, initial_state=[state_h, state_c]) # initial state of first decoder lstm is last encdoer state
        feature_dict[j] = x
        
    # if you use bidirectional, maybe outputs dims is not match inputs dims
    if x.shape[-1] != inputs_dims:
        outputs = CuDNNLSTM(units=n_dims, kernel_initializer='he_normal', return_sequences=True, return_state=True, name=f'layer_{j+1}')(x)
    else:
        outputs = x
        
    loss_dict = {}
    classifier_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=score_vector))
    recons_loss = tf.reduce_sum(tf.pow(outputs - feed_list[0], 2))
    loss = classifier_loss + recons_loss if classifier_loss is not None else recons_loss
    
    loss_dict['classifier_loss'] = classifier_loss
    loss_dict['recons_loss'] = recons_loss
    loss_dict['loss'] = loss
    
    return outputs, loss_dict, feature_dict