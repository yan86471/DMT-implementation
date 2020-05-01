import numpy as np
import tensorflow as tf
from keras import backend as K

from utils.helper import masking_func

@tf.function
def Reconsruction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

@tf.function
def Perceptual_loss(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)

@tf.function
def Identity_Makeup_Reconstruction_loss(identity_code, makeup_code, transfer_identity_code, transfer_makeup_code):
    identity_loss = tf.reduce_mean(tf.abs(identity_code[0] - transfer_identity_code[0]))
    makeup_loss = tf.reduce_mean(tf.abs(makeup_code[0] - transfer_makeup_code[0]))
    IMRL_loss = identity_loss + makeup_loss
    return tf.reduce_mean(IMRL_loss)

@tf.function
def Makeup_loss(y_true, y_pred_image, y_mask, classes):
    y_true_face, y_true_brow, y_true_eye, y_true_lip = y_true
    face_mask, brow_mask, eye_mask, lip_mask = y_mask

    
    y_pred_face = y_pred_image * face_mask
    face_loss = tf.reduce_mean((y_true_face - y_pred_face) ** 2)

    y_pred_brow = y_pred_image * brow_mask
    brow_loss = tf.reduce_mean((y_true_brow - y_pred_brow) ** 2) * 20

    y_pred_eye = y_pred_image * eye_mask
    eye_loss = tf.reduce_mean((y_true_eye - y_pred_eye) ** 2) * 10

    y_pred_lip = y_pred_image * lip_mask
    lip_loss = tf.reduce_mean((y_true_lip - y_pred_lip) ** 2) * 10
    return (face_loss + brow_loss + eye_loss + lip_loss)

@tf.function
def Total_Variation_loss(feature):
    left_loss = tf.reduce_mean(tf.abs(feature[:, 1:, :, :] - feature[:, :-1, :, :]))
    down_loss = tf.reduce_mean(tf.abs(feature[:, :, 1:, :] - feature[:, :, :-1, :]))
    TV_loss = left_loss + down_loss
    return TV_loss

@tf.function
def Adversarial_loss(dis_output):
    fake_loss = tf.reduce_mean(tf.square(dis_output[0]-1))
    transfer_loss = tf.reduce_mean(tf.square(dis_output[1]-1))
    return (fake_loss + transfer_loss)

@tf.function
def Discriminator_loss(real, fake):
    real_loss = tf.square(real-1)
    fake_loss1 = tf.square(fake[0])
    fake_loss2 = tf.square(fake[1])
    return tf.reduce_mean(real_loss + fake_loss1 + fake_loss2)

@tf.function
def Attention_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.abs(tf.clip_by_value(y_true, 0.0, 1.0)- y_pred))
    return loss

@tf.function
def KL_loss(mean, z_log_var):
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(mean) - tf.exp(z_log_var), axis = 1)
    return tf.reduce_mean(kl_loss)

