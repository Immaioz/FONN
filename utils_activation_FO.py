import tensorflow as tf
from keras import layers, constraints
import act_func_FO as fo

# ============== TRAINABLE ACTIVATION FUNCTIONS (per-layer parameters) ==============

class PerLayerRelu(layers.Layer):
    def __init__(self, alpha_init=0.0, **kwargs):
        super(PerLayerRelu, self).__init__(**kwargs)
        self.alpha_init = alpha_init
    
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
            constraint=constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)
        )
        super().build(input_shape)
    
    def call(self, inputs):
        return fo.ReLU_FO(inputs, self.alpha)


class PerLayerSigmoid(layers.Layer):
    def __init__(self, **kwargs):
        super(PerLayerSigmoid, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
            constraint=constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)
        )
        super().build(input_shape)
    
    def call(self, inputs):
        return fo.sigmoid_FO(inputs, self.alpha)


class PerLayerTanh(layers.Layer):
    def __init__(self, **kwargs):
        super(PerLayerTanh, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
            trainable=True,
            constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
        )
        super().build(input_shape)
    
    def call(self, inputs):
        return fo.tanh_FO(inputs, self.alpha) 


# ============== PerNeuron ACTIVATION (per-neuron trainable) ==============

class PerNeuronRelu(layers. Layer):
    def __init__(self, units, use_bias=True, seed=None, **kwargs):
        super(PerNeuronRelu, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.seed = seed
    
    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializer,
            trainable=True,
            name='kernel'
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
            name='alpha',
            constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)  # α ∈ [-1, 1]
        )
        super().build(input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.w)
        if self.use_bias:
            z = z + self.b
        return fo.ReLU_FO(z, self.alpha)
    
    def get_weights(self):
        if self.use_bias:
            return [self.w.numpy(), self.b.numpy(), self.alpha.numpy()]
        return [self.w.numpy(), self.alpha.numpy()]


class PerNeuronSigmoid(layers.Layer):
    def __init__(self, units, use_bias=True, seed=None, **kwargs):
        super(PerNeuronSigmoid, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.seed = seed
    
    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializer,
            trainable=True,
            name='kernel'
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
            name='alpha',
            constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)  # α ∈ [-1, 1]
        )
        super().build(input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.w)
        if self.use_bias:
            z = z + self.b
        alpha_safe = tf.maximum(tf.abs(self.alpha), 1e-7) * tf.sign(self.alpha + 1e-7)
        return fo.sigmoid_FO(z, alpha_safe)
    
    def get_weights(self):
        if self.use_bias: 
            return [self.w.numpy(), self.b.numpy(), self.alpha.numpy()]
        return [self.w.numpy(), self.alpha.numpy()]


class PerNeuronTanh(layers.Layer):
    def __init__(self, units, use_bias=True, seed=None, **kwargs):
        super(PerNeuronTanh, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.seed = seed
    
    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
        self.w = self.add_weight(
            shape=(input_shape[-1], self. units),
            initializer=initializer,
            trainable=True,
            name='kernel'
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
            trainable=True,
            name='alpha',
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)  # α ∈ [0, 1]
        )
        super().build(input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.w)
        if self.use_bias:
            z = z + self.b
        return fo.tanh_FO(z, self.alpha)
    
    def get_weights(self):
        if self.use_bias:
            return [self.w.numpy(), self.b.numpy(), self.alpha.numpy()]
        return [self.w.numpy(), self.alpha.numpy()]
