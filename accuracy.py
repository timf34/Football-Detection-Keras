from keras import backend as K
import tensorflow as tf

def deepball_precision(y_true, y_pred):
    ball_gt = y_true[:, :, :, 0]
    ball_cm = y_pred[:, :, :, 0]

    thre_ball_cm = K.cast(K.greater(ball_cm, 0.99), "float32")
    tp = K.sum(ball_gt * thre_ball_cm)
    totalp = K.sum(K.max(thre_ball_cm, axis=(1, 2)))
    prec = tp / tf.maximum(1.0, totalp)

    return prec

