import tensorflow as tf
import tensorflow_addons as tfa

kernel_reg = None#tf.keras.regularizers.l2(5e-5)

'''
Elementary Layer Definition
'''
def Conv2D_layer(x, filters, kernel_size, *args, **kwargs):
    default_kwargs = {'strides' : (1, 1), 'kernel_regularizer': kernel_reg, 'padding' : 'same', 'kernel_initializer' : tf.keras.initializers.he_normal()}
    default_kwargs.update(kwargs)

    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, *args, **default_kwargs)(x)
    return x

def UpSampling2D_layer(x, size, *args, **kwargs):
    default_kwargs = {}
    default_kwargs.update(kwargs)
    x = tf.keras.layers.UpSampling2D(size = size, *args, **default_kwargs)(x)
    return x

def Dense_layer(x, units, *args, **kwargs):
    default_kwargs = {'kernel_regularizer': kernel_reg, 'kernel_initializer' : tf.keras.initializers.he_normal()}
    default_kwargs.update(kwargs)

    x = tf.keras.layers.Dense(units, *args, **default_kwargs)(x)
    return x

def LayerNormalization_layer(x, *args, **kwargs):
    default_kwargs = {"axis" : 1, "center" : True, "scale" : True, "beta_initializer" : "zeros", "gamma_initializer" : "ones"}
    default_kwargs.update(kwargs)
    x = tf.keras.layers.LayerNormalization(*args, **kwargs)(x)
    return x

def InstanceNormalization_layer(x, *args, **kwargs):
    default_kwargs = {"axis" : 3,  "center" : True, "scale" : True, "beta_initializer" : "random_uniform", "gamma_initializer" : "random_uniform"}
    default_kwargs.update(kwargs)
    x = tfa.layers.InstanceNormalization(axis = 3, center = True, scale = True)(x)
    return x

def AdaInstanceNormalization_layer(content, style, epsilon=1e-5):
    style = tf.reshape(style, shape = (-1, 1, 1, style.shape[1]))
    meanC, varC = tf.nn.moments(content, [1, 2], keepdims=True)
    meanS, varS = tf.nn.moments(style,   [1, 2, 3], keepdims=True)
    sigmaC = tf.sqrt(tf.add(varC, epsilon))
    sigmaS = tf.sqrt(tf.add(varS, epsilon))
    return (content - meanC) * sigmaS / sigmaC + meanS

def ReLU_layer(x):
    x = tf.nn.relu(x)
    return x

def LeakyReLU_layer(x, alpha = 0.1):
    x = tf.keras.layers.LeakyReLU(alpha = alpha)(x)
    return x

def GlobalAveragePooling2D(x):
    x = tf.reduce_mean(x, axis=[1, 2])
    return x
