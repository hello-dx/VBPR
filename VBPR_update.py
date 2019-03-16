# An implement of VBPR using tensorflow #
# 2019.3.16 #
import numpy as np
import random
import json
import tensorflow as tf
import vgg16


def get_variable(type, shape, mean, stddev, name):
    if type == 'W':
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))
        tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
        return var
    elif type == 'b':
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                               initializer=tf.zeros_initializer())


class TVBPR:
    def __init__(self, K, K2):
        self.R = []  # Rating matrix
        self.nUsers = 0
        self.nItems = 0
        self.user_dict = {}
        self.item_dict = {}
        self.imageFeatures = {}
        self.imageFeaMatrix = []
        self.itemFeatureDim = 4096
        self.k = K  # Latent dimension
        self.k2 = K2  # Visual dimension
        self.MF_loss = 0


    def load_training_data(self, user_file, item_file):
        ## build item attribute dict

        with open(user_file,'r') as f:
            self.user_dict = json.load(f)
        with open(item_file,'r') as f:
            self.item_dict = json.load(f)

        self.nUsers = len(self.user_dict)
        self.nItems = len(self.item_dict)
        print('User: %d. Item: %d.'%(self.nUsers, self.nItems))

    def VBPR(self, user_idx, itemFea_pos, itemFea_neg, pos_item_idx, neg_item_idx):
        MF_U = get_variable(type='W', shape=[self.nUsers, self.k], mean=0, stddev=0.01, name='MF_U')
        MF_I = get_variable(type='W', shape=[self.nItems, self.k], mean=0, stddev=0.01, name='MF_I')
        visual_U = get_variable(type='W', shape=[self.nUsers, self.k2], mean=0, stddev=0.01,
                                name='visual_U')

        # MF_factor
        MF_U_factor = tf.gather(MF_U, user_idx)
        MF_U_factor = tf.reshape(MF_U_factor, shape=[1, tf.shape(MF_U_factor)[0]])  ##[1, k]
        MF_I_factor_pos = tf.gather(MF_I, pos_item_idx)  ## [?, nItem]
        MF_I_factor_neg = tf.gather(MF_I, neg_item_idx)  ## [?, nItem]

        # feature_factor
        visual_U_vector = tf.gather(visual_U, user_idx)
        visual_U_vector = tf.reshape(visual_U_vector, shape=[1, tf.shape(visual_U_vector)[0]])  ##[1, k2]


        itemEmb_W = get_variable(type='W', shape=[self.itemFeatureDim, self.k2],
                                 mean=0, stddev=0.01, name='itemEmb_W')
        itemEmb_b = get_variable(type='b', shape=[self.k2], mean=0, stddev=0.01, name='itemEmb_b')

        visual_U_factor = visual_U_vector
        visual_I_factor_pos = tf.sigmoid(tf.matmul(itemFea_pos, itemEmb_W) + itemEmb_b)
        visual_I_factor_neg = tf.sigmoid(tf.matmul(itemFea_neg, itemEmb_W) + itemEmb_b)


        BPR_user_factor = tf.concat([MF_U_factor, visual_U_factor], axis=1)
        BPR_item_factor_pos = tf.concat([MF_I_factor_pos, visual_I_factor_pos], axis=1)
        BPR_item_factor_neg = tf.concat([MF_I_factor_neg, visual_I_factor_neg], axis=1)

        uij = tf.multiply(BPR_user_factor, BPR_item_factor_pos)  # (?, concat_Dim)
        uik = tf.multiply(BPR_user_factor, BPR_item_factor_neg)  # (?, concat_Dim)
        self.uij = tf.sigmoid(tf.reshape(tf.reduce_sum(uij, axis=1), shape=[tf.shape(uij)[0], 1]))  # (?, 1)
        self.uik = tf.sigmoid(tf.reshape(tf.reduce_sum(uik, axis=1), shape=[tf.shape(uik)[0], 1]))  # (?, 1)

    def train(self):
        self.user_idx = tf.placeholder(dtype=tf.int32, shape=[])
        self.itemFea_pos = tf.placeholder(dtype=tf.float32, shape=[None, self.itemFeatureDim])
        self.itemFea_neg = tf.placeholder(dtype=tf.float32, shape=[None, self.itemFeatureDim])
        self.pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None, ])

        self.VBPR(self.user_idx, self.itemFea_pos, self.itemFea_neg, self.pos_item_idx, self.neg_item_idx)

        # train
        uij_shape = tf.shape(self.uij)[0]
        uik_shape = tf.shape(self.uik)[0]
        uij = tf.tile(self.uij, [uik_shape, 1])
        uik = tf.reshape(tf.tile(self.uik, [1, uij_shape]), shape=[-1, 1])

        BPR_loss = tf.reduce_mean(-tf.log(tf.sigmoid(uij - uik)))
        self.loss = BPR_loss  # + regular_loss
        self.accuracy = tf.reduce_sum(tf.round(tf.sigmoid(uij - uik))) / tf.reduce_sum(tf.ceil(tf.sigmoid(uij - uik)))

        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(0.01, self.global_step, decay_steps=5000, decay_rate=0.85,
                                                   staircase=False)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


