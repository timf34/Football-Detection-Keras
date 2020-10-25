import tensorflow as tf
import numpy as np
from constants import*
from keras import backend as K

def centernet_focal_loss(hm_true, hm_pred):
    #TODO: this is not working well...
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)


    pos_loss = -1*tf.math.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -1*tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss

def binary_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.
    alpha = .25
    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    y_pred = y_pred + epsilon
    # Clip the prediciton value
    #y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def deepball_loss_function(y_true, y_pred):

    ball_gt, bg_gt = y_true[:, :, :, 0], y_true[:, :, :, 1]
    N = K.sum(ball_gt, axis=(1, 2)) + 1
    M = K.sum(bg_gt, axis=(1, 2)) + 1
    zer = K.zeros_like(ball_gt)

    y_pred = K.log(y_pred)
    ball_cm = y_pred[:, :, :, 0]
    bg_cm = y_pred[:, :, :, 1]

    Lpos = K.sum(zer + (ball_cm * ball_gt), axis=(1, 2))
    Lpos = K.sum(K.zeros_like(N) + (Lpos / tf.maximum(1.0, N)))

    Lneg = K.sum(zer + (bg_cm * bg_gt), axis=(1, 2))
    Lneg = K.sum(K.zeros_like(M) + (Lneg / tf.maximum(1.0, M)))
    print (Lpos)
    print(K.eval(Lpos),K.eval(Lneg))

    # Multiplying by batch_size as Keras automatically averages the scalar output over it
    return (-Lpos - 0.3 * Lneg) * BATCH_SIZE


