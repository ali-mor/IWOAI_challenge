""" This script calls the model and data preparation for each plane """ 
""" and generate the output for each networks"""
import data_prep_test as data_preparation
import sys
import os
import numpy as np
import random
import model
import tensorflow as tf
####################################################
gpu_n = str(sys.argv[1])
path_img = str(sys.argv[2])
plane = str(sys.argv[3])
############## read images ###############
#directory = "experiment_01_enc_dec_adam/"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_n
############### hyper parameters ######################
##############################################################
n_filters = {}
filter_height = {}
filter_height["block_1"] = (6)*[3]
filter_height["block_2"] = (8)*[3]
filter_height["block_3"] = (11)*[3]
filter_height["block_4"] = (15)*[3]
filter_height["block_5"] = (20)*[3]
filter_height["block_6"] = (20)*[3]
filter_height["block_7"] = (15)*[3]
filter_height["block_8"] = (11)*[3]
filter_height["block_9"] = (8)*[3]
filter_height["block_10"] = (6)*[3]
filter_height["last_layer"] = 3
filter_width = filter_height
### max pooling
pooling = {}
pooling["block_1"] = 0
pooling["block_2"] = 0
pooling["block_3"] = 0
pooling["block_4"] = 0
### Activation function
activ_f = {}
activ_f["in_layer"] = 0
activ_f["block_1"] = (len(filter_height["block_1"]))*[0]
activ_f["block_2"] = (len(filter_height["block_2"]))*[0]
activ_f["block_3"] = (len(filter_height["block_3"]))*[0]
activ_f["block_4"] = (len(filter_height["block_4"]))*[0]
activ_f["block_5"] = (len(filter_height["block_5"]))*[0]
activ_f["block_6"] = (len(filter_height["block_6"]))*[0]
activ_f["block_7"] = (len(filter_height["block_7"]))*[0]
activ_f["block_8"] = (len(filter_height["block_8"]))*[0]
activ_f["block_9"] = (len(filter_height["block_9"]))*[0]
activ_f["block_10"] = (len(filter_height["block_10"]))*[0]
############### validation function ######################
##############################################################	
img = data_preparation.main(path_img, plane)
def gen_out_net():
    sh_img = img.shape
    sh_gt = (sh_img[0], sh_img[1], sh_img[2], 7)
    out_bin_n = np.zeros(sh_gt)
    valid_gt = np.zeros(sh_gt)
    for i in range(0, sh_gt[0], batch_size):
        out_bin_n[i: i+batch_size] = sess.run(out_bin, feed_dict=\
        {x: img[i:i+batch_size], y_: valid_gt[i:i+batch_size], is_train:False})
    if plane=='A1':
        out_bin_n = np.swapaxes(out_bin_n, 0, 1)
        out_bin_n = np.swapaxes(out_bin_n, 2, 1)
    if plane=='A2':
        out_bin_n = np.swapaxes(out_bin_n, 0, 1)
    np.save(path_img[:-3] + '_' + plane + '.npy', out_bin_n[..., 1:])
############### Read data ######################
##############################################################
sh_input = img.shape
in_shape = np.array([sh_input[1],sh_input[2]])
##############################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = False
sess = tf.Session(config=config)
net_model = model.Dense_net(in_shape, n_filters, filter_height, filter_width, \
pooling, activ_f)
output_prob, out_bin, ind_max, out_conv, loss, train_step, x, y_, is_train, metrics_values = \
net_model.build_model(gpu_n, 'Adam')
saver = tf.train.Saver(max_to_keep = 0)
saver.restore(sess, "models/ep_"+ plane + ".ckpt")
batch_size = 4
gen_out_net()
