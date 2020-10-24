import tensorflow as tf
import numpy as np
from keras import backend as K
from loss import*

kvar1 = tf.keras.backend.variable(np.array([[[False, False, False], [False, False, False], [False, False, False]]]),
                                 dtype='bool')
kvar2 = tf.keras.backend.variable(np.array([[[[0, 1, 1], [1, 0,0]]]]),
                                 dtype='float32')
gt = tf.keras.backend.variable(np.array([[[[0, 0, 1], [0, 0,0]]]]),
                                 dtype='float32')

x=deepball_loss_function(gt, kvar2)






"""c= K.cast(K.greater(kvar2, 0.99), 'float32')
print ('c')
print(c)
g = K.sum(gt * kvar2)
print('g')
print (g)
totalp = K.max(c, axis=(1, 2))
print('tot')
print(totalp)

prec = g / tf.maximum(1.0, totalp)
print(prec)"""