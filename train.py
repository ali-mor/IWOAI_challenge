""" This script calls the model and data preparation for each plane """ 
""" and train the networks for each plane"""
import data_prep_train as data_preparation
import sys
import os
import numpy as np
import random
import model
import tensorflow as tf
####################################################
gpu_n = str(sys.argv[1])
directory = str(sys.argv[2])
path_data = str(sys.argv[3])
plane = str(sys.argv[4])
############## read images ###############
#directory = "experiment_01_enc_dec_adam/"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_n
directory = directory + '/' + plane
if not os.path.exists(directory):
	os.makedirs(directory)
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
### 0=max
pooling = {}
pooling["block_1"] = 0
pooling["block_2"] = 0
pooling["block_3"] = 0
pooling["block_4"] = 0
### 0=relu
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
loss_va_all = []
loss_va_all_it = []
metric_va_all = []
metric_va_all_it = []
def valid_cost(ep):
	metrics = []
	L_v_all = 0.
	all_keys = valid_image.keys()
	metric_per = np.zeros((6,4))
	for k in all_keys:
		output_prob_np = np.zeros(valid_gt[k].shape)
		for i in range(0, output_prob_np.shape[0], batch_size):
			output_prob_np[i:i+batch_size] = sess.run(out_bin, feed_dict=\
			{x: valid_image[k][i:i+batch_size], \
			y_: valid_gt[k][i:i+batch_size], is_train:False})
		if plane=='A1':
			output_prob_np = np.swapaxes(output_prob_np, 0, 1)
			output_prob_np = np.swapaxes(output_prob_np, 2, 1)
			valid_gt[k] = np.swapaxes(valid_gt[k], 0, 1)
			valid_gt[k] = np.swapaxes(valid_gt[k], 2, 1)
		if plane=='A2':
			output_prob_np = np.swapaxes(output_prob_np, 0, 1)
			valid_gt[k] = np.swapaxes(valid_gt[k], 0, 1)
		for nn in range(6):
			class_met = model.metrics(output_prob_np[..., nn+1], valid_gt[k][..., nn+1])
			metric_per[nn, :] = metric_per[nn, :] + class_met.metrics_cal_cpu()
	metric_per = metric_per/(len(all_keys))
	print("Ave. Dice=", np.mean(metric_per, axis=0)[3])
	save(output_prob_np, directory+'/out_prob_'+ str(ep).zfill(3)+'_'+str(k)+'.nii')
############### Read data ######################
##############################################################
train_image, train_gt, valid_image, valid_gt = \
data_preparation.main(path_data, plane)
sh_input = train_image.shape
in_shape = np.array([sh_input[1],sh_input[2]])
################ Config the graph options #####################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = False
sess = tf.Session(config=config)
tf.set_random_seed(123)
net_model = model.Dense_net(in_shape, n_filters, filter_height, filter_width, \
pooling, activ_f)
output_prob, out_bin, ind_max, out_conv, loss, train_step, x, y_, is_train, metrics_values = \
net_model.build_model(gpu_n, 'Adam')
saver = tf.train.Saver(max_to_keep = 0)
sess.run(tf.global_variables_initializer())
ss = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Number of parameters:", ss)
########### Traning parameters ###################
batch_size = 4
iteration = len(train_image)/batch_size
print ('Number of iterations:', iteration)
random.seed(1)
loss_tr_all = []
loss_tr_all_it = []
num_epoch = 150
###########Start training ###################
for ep in range(num_epoch):
	print ("epoch=", ep+1)
	sh_train = random.sample(range(sh_input[0]), sh_input[0])
	L_t_all = 0.
	for n_itr in range(int(iteration)):
		batch_xs = train_image[sh_train[batch_size*n_itr:(n_itr+1)*batch_size]]
		batch_xt = train_gt[sh_train[batch_size*n_itr:(n_itr+1)*batch_size]]
		_, L_t = sess.run((train_step, loss), feed_dict={x: batch_xs, y_: batch_xt})
		L_t_all = L_t_all + L_t 
		loss_tr_all_it.append(L_t)	
	L_t_all = L_t_all / int(iteration)
	loss_tr_all.append(L_t_all)
	print('train loss:', L_t_all)
	directory2 = directory + "/models/"
	if not os.path.exists(directory2):
		os.makedirs(directory2)
	saver.save(sess, directory2+"ep_" + str(ep+1).zfill(3)+'.ckpt')
	if ep>-1:
		valid_cost(ep+1) 
	

	
