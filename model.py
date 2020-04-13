""" This script generates deep learning model"""
import tensorflow as tf
import numpy as np
class layers(object):
    """define layers used in the network architecture"""
    def __init__(self, x_in, ch_in, ch_out):
        self.x_in = x_in
        self.ch_in = ch_in
        self.ch_out = ch_out
    def weight_variable(self,shape):
        """defien variable for network weights"""
        initializer = tf.initializers.he_uniform(seed=123)
        return tf.Variable(initializer(shape))
    def bias_variable(self,shape):
        """define variables for bias"""
        return tf.Variable(tf.constant(0.1, shape=shape), name="bias")
    def conv2d(self,x, W):
        """ define 2D convolution"""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name="conv")
    def pool_2x2(self, pooling):
        """define 2D max pooling layer"""
        return tf.nn.max_pool(self.x_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    def unpooling_2x2(self):
        """define theb bilinear interpolation function for upsampling"""
        with tf.name_scope('unpooling') as scope:
            return tf.image.resize_images(self.x_in, [self.ch_in, self.ch_out], method=1)
    def pad_con_batch_relu(self, filter_height, filter_width, activ_f, is_train):
        """ define zero padding+2D convolution+batch normalizarion+relu"""
        with tf.name_scope('batch_relu_conv') as scope:
            p_s_h = int(filter_height/2)
            p_s_w = int(filter_width/2)
            epsilon = 1e-3
            ### zero padding
            self.x_padded = tf.pad(self.x_in, [[0, 0], [p_s_h, p_s_h], [p_s_w, p_s_w], [0, 0]], "CONSTANT")
            ### conv layer
            self.h_conv = self.conv2d(self.x_padded, self.weight_variable(\
            [filter_height, filter_width, self.ch_in, self.ch_out])) + \
            self.bias_variable([self.ch_out])
            ### batch normalization
            self.batch_mean, self.batch_var = tf.nn.moments(self.h_conv, [0, 1, 2])
            h1 = tf.nn.batch_normalization(self.h_conv, self.batch_mean, self.batch_var,\
            tf.Variable(tf.zeros([self.ch_out])), tf.Variable(tf.ones([self.ch_out])), epsilon)
            ### activation function
            return tf.nn.relu(h1)
    def out_layer(self, filter_height, filter_width):
        """ define the last layer in network: conv.+ softmax """
        with tf.name_scope('out_layer') as scope:
            p_s_h = int(filter_height/2)
            p_s_w = int(filter_width/2)
            self.x_padded=tf.pad(self.x_in,[[0,0],[p_s_h, p_s_h],[p_s_w, p_s_w],[0,0]],"CONSTANT")
            ### conv layer
            self.h_conv = self.conv2d(self.x_padded, self.weight_variable([filter_height, filter_width, self.ch_in, self.ch_out])) + self.bias_variable([self.ch_out])
            out_prob = tf.nn.softmax(self.h_conv)
            ind_max = tf.argmax(out_prob, axis=-1)
            out_bin = tf.one_hot(ind_max, depth=7)
            return self.h_conv, out_prob, out_bin, ind_max
    def input_layer(self, is_train, activ_f):
        """ Define input layer of the netwotk: conv. layer with 7x7 kernel size"""
        p_s=3
        filter_hight=7
        filter_width=7
        epsilon = 1e-3
        ### zero padding
        self.x_padded = tf.pad(self.x_in, [[0, 0], [p_s, p_s], [p_s, p_s], [0, 0]], "CONSTANT")
        ### conv layer
        self.h_conv = self.conv2d(self.x_padded, self.weight_variable([filter_hight, \
        filter_width, self.ch_in, self.ch_out])) + self.bias_variable([self.ch_out])
        ### batch normalization
        self.batch_mean, self.batch_var = tf.nn.moments(self.h_conv, [0, 1, 2])
        h1 = tf.nn.batch_normalization(self.h_conv, self.batch_mean, self.batch_var,\
        tf.Variable(tf.zeros([self.ch_out])), tf.Variable(tf.ones([self.ch_out])), epsilon)
        ### activation function
        return tf.nn.relu(h1)		
        
    def cost_ce(self, y_):
        """define loss function as cross entropy for classification task"""
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
        (labels=y_, logits=self.x_in), name="CE_loss")

    def optimizer(self,loss,optimizer_name):
        """ define optimizers as Adam"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, \
            beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss)

class dense_block_2(object):
    """ a block of CNNs: (upsmapling layer: for decoder part)+ N pad+CNN+BN+reLu """
    """ layers (followed by max pooling: for encoder part) """
    def __init__ (self, x_in, ch1, ch2, pool_flag, in_shape, filter_height, \
    filter_width, pooling, activ_f):
        self.x_in = x_in
        self.ch1 = ch1
        self.ch2 = ch2
        self.pool_flag = pool_flag
        self.in_shape = in_shape
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.pooling = pooling
        self.activ_f = activ_f

    def instant(self, is_train):
        with tf.name_scope('Unpooling'):
            x_t = {}
            x_t[-1] = self.x_in
            if self.pool_flag == 2:
                unpool = layers(self.x_in, self.in_shape[0], self.in_shape[1])	
                x_t[-1] = unpool.unpooling_2x2()
        with tf.name_scope('Dense_Block'):
            N_layers = len(self.filter_height)
            for n_layer in range(N_layers):
                layer_t = layers(x_t[-1], self.ch1+self.ch2*n_layer, self.ch2)
                x_t[n_layer] = layer_t.pad_con_batch_relu(self.filter_height[n_layer], \
                self.filter_width[n_layer], self.activ_f[n_layer], is_train)
                if n_layer!=N_layers-1:
                    x_t[-1] = tf.concat([x_t[-1], x_t[n_layer]], axis=-1)
        if self.pool_flag==2 or self.pool_flag==0:
            return x_t[n_layer]
        if self.pool_flag==1:
            with tf.name_scope('Pooling'):
                C = 2
                concat4 = tf.concat([x_t[-1], x_t[N_layers-1]], axis=-1)
                layer_t = layers(concat4, self.ch1+self.ch2*N_layers, \
                (self.ch1+self.ch2*N_layers)/C)
                h_conv = layer_t.pad_con_batch_relu(1, 1, 0, is_train)
                max_pool=layers(h_conv, 0, 0)	
                self.h_pool = max_pool.pool_2x2(self.pooling)
                return self.h_pool

class metrics(object):
    """ calculates meterics """
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual
    def metrics_cal(self):
        """calculate recall, precision, and ... in GPU"""
        TP = tf.count_nonzero(self.predicted * self.actual, dtype=tf.float32, axis=[0, 1, 2])
        TN = tf.count_nonzero((self.predicted - 1) * (self.actual - 1), dtype=tf.float32, axis=[0, 1, 2])
        FP = tf.count_nonzero(self.predicted * (self.actual - 1), dtype=tf.float32, axis=[0, 1, 2])
        FN = tf.count_nonzero((self.predicted - 1) * self.actual, dtype=tf.float32, axis=[0, 1, 2])
        eps = tf.constant(1e-6,dtype=tf.float32)
        recall = TP/(TP+FN+eps)
        specificity = TN/(TN+FP+eps)
        precision = TP/(TP+FP+eps)
        dice = (recall*precision*2.) / (recall+precision+eps)
        return [recall,specificity,precision,dice]
    def metrics_cal_cpu(self):
        """calculate recall, precision, and ... in CPU"""
        TP = np.sum(self.predicted[self.actual==1])
        FP = np.sum(self.predicted[self.actual==0])
        FN = np.sum(self.actual[self.predicted==0])
        TN = np.sum(1 - (self.predicted[self.actual==0]))
        eps = 1e-6
        recall = float(TP)/(TP+FN+eps)
        specificity = float(TN)/(TN+FP+eps)
        precision = float(TP)/(TP+FP+eps)
        #accuracy=float(TP+TN)/(TP+FP+TN+FN+0.00001)
        dice = (recall * precision * 2.) / (recall + precision+eps)
        return [recall, specificity, precision, dice]
			
class Dense_net(object):
    """ create architecture as densely connected network"""
    def __init__(self, in_shape, n_filters, filter_height, filter_width, pooling, activ_f):
        """ initial parmaters"""
        self.in_shape = in_shape
        self.n_filters = n_filters
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.pooling = pooling		
        self.activ_f = activ_f		
    def build_model(self,gpu_n,optimizer_name):
        """ build network model"""
        with tf.name_scope('network') as scope:
            ###############Variables######################
            ##############################################################
            x = tf.placeholder(tf.float32, shape=[None, self.in_shape[0], \
            self.in_shape[1], 1])
            y_ = tf.placeholder(tf.float32, shape=[None, self.in_shape[0], \
            self.in_shape[1], 7])
            is_train = tf.placeholder_with_default(True, shape=())
            tf.set_random_seed(123)
            GR = 12
            in_f = GR
            C = 2
            ###############Input Layer######################
            ##############################################################
            layer_in = layers(x, 1, in_f)
            layer_in_o = layer_in.input_layer(is_train, self.activ_f["in_layer"])
            ###############First block######################
            ##############################################################
            block_1 = dense_block_2(layer_in_o, in_f, GR, 1, self.in_shape, \
            self.filter_height["block_1"], self.filter_width["block_1"], \
            self.pooling["block_1"], self.activ_f["block_1"])
            h_pool1 = block_1.instant(is_train)
            ###############Second block######################
            ##############################################################
            in_ch2 = (in_f + GR * len(self.filter_height["block_1"])) / C
            block_2 = dense_block_2(h_pool1, in_ch2, GR, 1, \
            self.in_shape, self.filter_height["block_2"], self.filter_width["block_2"], \
            self.pooling["block_2"], self.activ_f["block_2"])
            h_pool2 = block_2.instant(is_train)
            ###############3th block######################
            ##############################################################
            in_ch3 = (in_ch2 + GR * len(self.filter_height["block_2"])) / C
            block_3 = dense_block_2(h_pool2, in_ch3, GR, 1, \
            self.in_shape, self.filter_height["block_3"], self.filter_width["block_3"], \
            self.pooling["block_3"], self.activ_f["block_3"])
            h_pool3 = block_3.instant(is_train)
            ###############4th block######################
            ##############################################################
            in_ch4 = (in_ch3 + GR * len(self.filter_height["block_3"])) / C
            block_4 = dense_block_2(h_pool3, in_ch4, GR, 1, \
            self.in_shape, self.filter_height["block_4"], self.filter_width["block_4"], \
            self.pooling["block_4"], self.activ_f["block_4"])
            h_pool4 = block_4.instant(is_train)
            ###############5th block######################
            ##############################################################
            in_ch5 = (in_ch4 + GR * len(self.filter_height["block_4"])) / C
            block_5 = dense_block_2(h_pool4, in_ch5, GR, 0, \
            self.in_shape, self.filter_height["block_5"], self.filter_width["block_5"], \
            -1, self.activ_f["block_5"])
            rel1 = block_5.instant(is_train)
            ###############6th block######################
            ##############################################################
            block_6 = dense_block_2(rel1,  GR, GR, 0, \
            self.in_shape, self.filter_height["block_6"], self.filter_width["block_6"], \
            -1, self.activ_f["block_6"])
            rel2 = block_6.instant(is_train)
            ###############7th block######################
            ##############################################################
            concat_1 = tf.concat([h_pool4, rel2], axis=-1)
            block_7 = dense_block_2(concat_1, in_ch5+GR, GR, 2, np.int32(np.ceil(self.in_shape/8.)), \
            self.filter_height["block_7"], self.filter_width["block_7"], \
            -1, self.activ_f["block_7"])
            unpool1 = block_7.instant(is_train)
            ###############8th block######################
            ##############################################################
            concat_2 = tf.concat([h_pool3, unpool1], axis=-1)
            block_8 = dense_block_2(concat_2, in_ch4+GR, GR, 2, np.int32(np.ceil(self.in_shape/4.)), \
            self.filter_height["block_8"], self.filter_width["block_8"], \
            -1, self.activ_f["block_8"])
            unpool2 = block_8.instant(is_train)
            ###############9th block######################
            ##############################################################
            concat_3 = tf.concat([h_pool2, unpool2], axis=-1)
            block_9 = dense_block_2(concat_3,in_ch3+GR, GR, 2, np.int32(np.ceil(self.in_shape/2.)), \
            self.filter_height["block_9"], self.filter_width["block_9"], \
            -1, self.activ_f["block_9"])
            unpool3 = block_9.instant(is_train)
            ###############10th block######################
            ##############################################################
            concat_4 = tf.concat([h_pool1, unpool3], axis=-1)
            block_10 = dense_block_2(concat_4, in_ch2+GR, GR, 2, \
            self.in_shape, self.filter_height["block_10"], self.filter_width["block_10"], \
            -1, self.activ_f["block_10"])
            unpool4 = block_10.instant(is_train)
            ###############Output block######################
            ####################################################################
            out = layers(unpool4, GR, 7)
            out_conv, output_prob, out_bin, ind_max = out.out_layer(\
            self.filter_height["last_layer"], self.filter_width["last_layer"])
            ###############Cost function block######################
            ####################################################################
            cost_f = layers(out_conv, 0, 0)
            loss = cost_f.cost_ce(y_)
            ###############optimizer ######################
            ####################################################################
            train_step = cost_f.optimizer(loss, optimizer_name)
        return output_prob, out_bin, ind_max, out_conv, loss, train_step, x, y_, \
        is_train, 0

