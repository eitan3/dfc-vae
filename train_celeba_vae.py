import mxnet as mx
import numpy as np
from mxnet import nd
from vae_module import train_vae
from vae_data import GetData
from config_celeba_cnn import dataset_path, batch_size, num_epochs, num_latent_space, optimizer, learning_rate, wd, beta1
from config_celeba_cnn import num_fc_hidden_units, img_size, log_interval, model_export_folder, prefix
from vae_symbols.simple_encoder import Encoder_Module
from vae_symbols.simple_decoder import Decoder_Module
from vae_symbols.simple_dfc import DFC_Module
import cv2
import os

# Initialize numpy as mxnet, set working device (ctx) as gpu 0
mx.random.seed(1)
np.random.seed(1)
ctx = mx.gpu(0)

# Get data and split to train and test
train_iter, test_iter = GetData(ctx, dataset_path)


# Evaluate model accuracy function
def evaluate_accuracy(data, net, ctx):
    numerator = 0.
    denominator = 0.
    data = data.as_in_context(ctx)
    out, mu, logvar, dfc_out = net(data)
    numerator += nd.mean(nd.abs(data - out))
    denominator += data.shape[0]
    return numerator / denominator


# Save true and predicted images
def save_images(save_folder, data_imgs, pred_imgs):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(batch_size):
        data_img = data_imgs[i]
        data_img = np.swapaxes(data_img, 0, 2)
        data_img = np.swapaxes(data_img, 1, 0)
        data_img = (data_img + 1.0) / 2.0 * 255.0
        data_img = data_img.astype(np.uint8)
        data_img = cv2.resize(data_img, (img_size * 3, img_size * 3))
        cv2.imwrite(save_folder + str(i) + '_1_input.jpg', data_img)

        pred_img = pred_imgs[i]
        pred_img = np.swapaxes(pred_img, 0, 2)
        pred_img = np.swapaxes(pred_img, 1, 0)
        pred_img = (pred_img + 1.0) / 2.0 * 255.0
        pred_img = pred_img.astype(np.uint8)
        pred_img = cv2.resize(pred_img, (img_size * 3, img_size * 3))
        cv2.imwrite(save_folder + str(i) + '_2_pred.jpg', pred_img)


# Reconstruct loss
def recon_loss(recon_x, x):
    def berhu_loss(Y, pred, batch_size, image_size):
        diff = Y - pred
        loss = nd.abs(diff)
        c = 0.2 * nd.max(loss)
        c = nd.broadcast_to(data=nd.reshape(data=c, shape=(1, 1)),
                            shape=(batch_size, image_size))
        loss = ((loss > c) * ((nd.square(diff) + nd.square(c)) / c * 2) +
                (loss <= c) * loss)
        loss = nd.mean(loss, axis=1)
        return loss

    def gradient_difference_loss(Y_true, Y_pred, alpha, shift=1):
        def slice_axis(net, axis, begin, end):
            net = nd.slice_axis(net, axis=axis, begin=begin, end=end)
            return net

        Y_true = nd.pad(Y_true, mode="edge", pad_width=(0, 0, 0, 0, shift, shift, shift, shift))  # , constant_value=0
        Y_pred = nd.pad(Y_pred, mode="edge", pad_width=(0, 0, 0, 0, shift, shift, shift, shift))  # , constant_value=0
        t1_true_slice_1 = slice_axis(Y_true, 2, shift, None)
        t1_true_slice_2 = slice_axis(Y_true, 2, 0, -1 * shift)
        t1_pred_slice_1 = slice_axis(Y_pred, 2, shift, None)
        t1_pred_slice_2 = slice_axis(Y_pred, 2, 0, -1 * shift)
        t2_true_slice_1 = slice_axis(Y_true, 3, 0, -1 * shift)
        t2_true_slice_2 = slice_axis(Y_true, 3, shift, None)
        t2_pred_slice_1 = slice_axis(Y_pred, 3, 0, -1 * shift)
        t2_pred_slice_2 = slice_axis(Y_pred, 3, shift, None)
        t1 = nd.power((t1_true_slice_1 - t1_true_slice_2) -
                      (t1_pred_slice_1 - t1_pred_slice_2), alpha)
        t2 = nd.power((t2_true_slice_1 - t2_true_slice_2) -
                      (t2_pred_slice_1 - t2_pred_slice_2), alpha)
        out = (nd.mean(nd.flatten(t1), axis=1) + nd.mean(nd.flatten(t2), axis=1)) * 0.5
        return out

    pred = nd.identity(data=recon_x)
    label = nd.identity(data=x)
    y_true_f = nd.flatten(label)
    y_pred_f = nd.flatten(pred)

    gd_scale = 0.25
    gd_loss1 = gradient_difference_loss(label, pred, 2)
    gd_loss2 = gradient_difference_loss(label, pred, 2, shift=2)
    gd_loss = (gd_loss1 + gd_loss2) * 0.5 * gd_scale

    mse_loss = nd.mean(nd.square(y_true_f - y_pred_f), axis=1)

    total_loss = gd_loss + mse_loss
    return total_loss


# Save folder
param_save_path = model_export_folder

# Set args
encoder_args = {'num_fc_hidden_units': num_fc_hidden_units,
                'num_latent_space': num_latent_space}
decoder_args = {'num_fc_hidden_units': num_fc_hidden_units,
                'num_latent_space': num_latent_space}

optimizer_args = {'wd': wd,
                  'beta1': beta1}
args = {'batch_size': batch_size,
        'epochs': num_epochs,
        'latent_space': num_latent_space,
        'optimizer': optimizer,
        'optimizer_args': optimizer_args,
        'lr': learning_rate,
        'log_interval': log_interval,
        'param_save_path': model_export_folder,
        'encoder_args': encoder_args,
        'decoder_args': decoder_args}

beta_args = {'KL_scale': 1.0 / float(num_latent_space),
             'gamma_minimum': 1.0,
             'gamma_maximum': 1.0,
             'gamma_change_duration': 500000,
             'capacity_maximum': 20,
             'capacity_change_duration': 500000,
             'start_moving_vars_epoch': 10,
             'update_vars_every_X_epoch': 5}

# Start training function
train_vae(ctx, prefix, train_iter, test_iter, Encoder_Module, Decoder_Module, DFC_Module, recon_loss, evaluate_accuracy,
          save_images, beta_args, args)