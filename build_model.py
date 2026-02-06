import tensorflow as tf
from tensorflow import keras
from keras import layers, constraints
import act_func_FO as fo
import utils_activation_FO as uaf


# ============== MODEL BUILDERS ==============
def fixed(num_input, num_hidden, num_output, activation, seed=None):
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    model = keras.Sequential([
        layers.Input(shape=(num_input,)),
        layers.Dense(num_hidden, 
                    kernel_initializer=initializer),
        layers.Activation(activation) if isinstance(activation, str) else activation,
        layers.Dense(num_hidden, kernel_initializer=initializer),
        layers.Activation(activation) if isinstance(activation, str) else 
            activation.__class__(), 
        layers.Dense(num_output, activation='linear', kernel_initializer=initializer)
    ])
    return model



def per_layer(num_input, num_hidden, num_output, activation, seed=None):
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    model = keras.Sequential([
        layers.Input(shape=(num_input,)),
        layers.Dense(num_hidden, 
                    kernel_initializer=initializer),
        layers.Activation(activation),
        layers.Dense(num_hidden, kernel_initializer=initializer),
        layers.Activation(activation),
        layers.Dense(num_output, activation='linear', kernel_initializer=initializer)
    ])
    return model

def per_neuron(num_input, num_hidden, num_output, bias=True, activation='Relu', seed=None):  
    inputs = keras.Input(shape=(num_input,))
    
    if activation == 'Relu':
        x = uaf.PerNeuronRelu(num_hidden, use_bias=bias, seed=seed)(inputs)
        x = uaf.PerNeuronRelu(num_hidden, use_bias=bias, seed=seed)(x)
    elif activation == 'Sigmoid':
        x = uaf.PerNeuronSigmoid(num_hidden, use_bias=bias, seed=seed)(inputs)
        x = uaf.PerNeuronSigmoid(num_hidden, use_bias=bias, seed=seed)(x)
    elif activation == 'Tanh':
        x = uaf.PerNeuronTanh(num_hidden, use_bias=bias, seed=seed)(inputs)
        x = uaf.PerNeuronTanh(num_hidden, use_bias=bias, seed=seed)(x)
    
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    outputs = layers.Dense(num_output, activation='linear', 
                          kernel_initializer=initializer)(x)
    
    return keras.Model(inputs, outputs)