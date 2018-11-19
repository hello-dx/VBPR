# An implement of VBPR using tensorflow #
# 2018.11.19 #

import csv
import numpy as np
import random
import json
import tensorflow as tf


class Processing:
    def __init__(self, K, K2):
        self.R = []  # Rating matrix
        self.nUsers = 0
        self.nItems = 0
        self.user_dict = {}
        self.item_dict = {}
        self.imageFeatures = {}
        self.imageFeaMatrix = []
        self.imageFeatureDim = 4096
        self.k = K   # Latent dimension
        self.k2 = K2  # Visual dimension
        self.MF_loss = 0

    # def load_data(self, image_feature_path, rating_file_path):
    #     # self.load_image_feature(image_feature_path)
    #     self.load_training_data(rating_file_path)

    def load_image_feature(self, image_feature_path):
        csv_reader=csv.reader(open(image_feature_path))
        for item in csv_reader:
            item_id=item[0]
            item_feature = item[1:]
            item_feature = list(map(float, item_feature))
            self.imageFeatures[item_id] = item_feature
        self.imageFeaMatrix=[[0.]*self.imageFeatureDim]*self.nItems
        for item in self.imageFeatures:
            try:
                self.imageFeaMatrix[self.item_dict[item]] = self.imageFeatures[item]
            except:
                pass

    def load_training_data(self, rating_file_path):
        with open(rating_file_path,'r') as f:
            data = json.load(f)

        # create user/item idx dictionary
        for user_id in data:
            if user_id not in self.user_dict.keys():
                self.user_dict[user_id] = self.nUsers
                self.nUsers += 1
                for item_id in data[user_id]:
                    if item_id not in self.item_dict.keys():
                        self.item_dict[item_id] = self.nItems
                        self.nItems += 1
        self.R = np.array([[0.] * self.nItems] * self.nUsers)
        for user_id in data:
            for item_id in data[user_id]:
                self.R[self.user_dict[user_id], self.item_dict[item_id]] = 10 #data[user_id][item_id][1]

def get_variable(type, shape, mean, stddev, name):
    if type == 'W':
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))
        tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
        return var
    elif type == 'b':
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                               initializer=tf.zeros_initializer())


def VBPR(itemFea_matrix, userFea_matrix, user_idx, pos_item_idx, neg_item_idx):
    MF_U = get_variable(type='W', shape=[model.nUsers, model.k], mean=0, stddev=0.01, name='MF_U')
    MF_I = get_variable(type='W', shape=[model.nItems, model.k], mean=0, stddev=0.01, name='MF_I')
    visual_U = get_variable(type='W', shape=[model.nUsers, model.k2], mean=0, stddev=0.01, name='visual_U')
    visual_I = itemFea_matrix

    MF_U_factor = tf.gather(MF_U, user_idx)
    MF_U_factor = tf.reshape(MF_U_factor, shape=[1, tf.shape(MF_U_factor)[0]])  ##[1, k]
    MF_I_factor_pos = tf.gather(MF_I, pos_item_idx)   ## [?, nItem]
    MF_I_factor_neg = tf.gather(MF_I, neg_item_idx)   ## [?, nItem]

    visual_U_vector = tf.gather(visual_U, user_idx)
    visual_U_vector = tf.reshape(visual_U_vector, shape=[1, tf.shape(visual_U_vector)[0]])  ##[1, k2]
    visual_I_matrix_pos = tf.gather(visual_I, pos_item_idx)
    visual_I_matrix_neg = tf.gather(visual_I, neg_item_idx)

    itemEmb_W = get_variable(type='W', shape=[model.imageFeatureDim, model.k2], mean=0, stddev=0.01, name='itemEmb_W')
    itemEmb_b = get_variable(type='b', shape=[model.k2], mean=0, stddev=0.01, name='itemEmb_b')

    visual_U_factor = visual_U_vector
    visual_I_factor_pos = tf.sigmoid(tf.matmul(visual_I_matrix_pos, itemEmb_W) + itemEmb_b)
    visual_I_factor_neg = tf.sigmoid(tf.matmul(visual_I_matrix_neg, itemEmb_W) + itemEmb_b)

    BPR_user_factor = tf.concat([MF_U_factor, visual_U_factor], axis=1)
    BPR_item_factor_pos = tf.concat([MF_I_factor_pos, visual_I_factor_pos], axis=1)
    BPR_item_factor_neg = tf.concat([MF_I_factor_neg, visual_I_factor_neg], axis=1)

    uij = tf.multiply(BPR_user_factor, BPR_item_factor_pos)  # (?, concat_Dim)
    uik = tf.multiply(BPR_user_factor, BPR_item_factor_neg)  # (?, concat_Dim)
    uij = tf.sigmoid(tf.reshape(tf.reduce_sum(uij, axis=1), shape=[tf.shape(uij)[0], 1]))  # (?, 1)
    uik = tf.sigmoid(tf.reshape(tf.reduce_sum(uik, axis=1), shape=[tf.shape(uik)[0], 1]))  # (?, 1)

    uij_shape = tf.shape(uij)[0]
    uik_shape = tf.shape(uik)[0]
    uij = tf.tile(uij, [uik_shape, 1])
    uik = tf.reshape(tf.tile(uik, [1, uij_shape]), shape=[-1, 1])

    BPR_loss = tf.reduce_mean(-tf.log(tf.sigmoid(uij - uik)))
    return BPR_loss, uij, uik



model=Processing(K=64, K2=128)
model.load_training_data('H:/heyFighting/my_code/fashion_shape/codes/preference/TEST/data_training.json')
model.load_image_feature('H:/heyFighting/my_code/fashion_shape/codes/preference/TEST/image_features.csv')

itemFea_matrix = tf.placeholder(dtype=tf.float32, shape=[model.nItems, model.imageFeatureDim])
userFea_matrix = tf.placeholder(dtype=tf.float32, shape=[None, model.nItems])
user_idx = tf.placeholder(dtype=tf.int32, shape=[])
pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None,])
neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None,])


BPR_loss, uij, uik = VBPR(itemFea_matrix, userFea_matrix, user_idx, pos_item_idx, neg_item_idx)
tf.add_to_collection('BPR_losses', BPR_loss)
regular_loss = tf.add_n(tf.get_collection('regular_losses'))
loss = BPR_loss #+ regular_loss

accuracy = tf.reduce_sum(tf.round(tf.sigmoid(uij - uik)))/tf.reduce_sum(tf.ceil(tf.sigmoid(uij - uik)))

global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps=1000, decay_rate=0.85, staircase=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)

max_acc = 0
for step in range(20001):
    user_i = random.randint(0, model.nUsers-1)
    step_now = step
    #create positive & negative item
    pos_item = [x for x in range(model.nItems) if model.R[user_i][x] != 0]
    neg_item = []
    while len(neg_item)< 5*len(pos_item):
        a=random.randint(0, model.nItems-1)
        if a not in pos_item:
            neg_item.append(a)

    _, los, B_loss, l2_loss, acc, uijj= sess.run([train_step, loss, BPR_loss, regular_loss, accuracy, uij],
                          feed_dict={itemFea_matrix: model.imageFeaMatrix,
                                     userFea_matrix: model.R,
                                     user_idx: user_i,
                                     pos_item_idx: pos_item,
                                     neg_item_idx: neg_item,
                                     global_step: step_now})

    print('step-%d, loss : %f(%f %f). Accuracy : %f'%(step_now, los, B_loss, l2_loss, acc))
    if step_now % 1000 == 0 and step_now>0:
        final_acc = 0
        for i in range(model.nUsers):
            valid_pos_item = [x for x in range(model.nItems) if model.R[i][x] != 0]
            valid_neg_item = []
            while len(valid_neg_item) < 5 * len(valid_pos_item):
                a = random.randint(0, model.nItems - 1)
                if a not in valid_pos_item:
                    valid_neg_item.append(a)

            valid_los, valid_acc = sess.run([loss, accuracy],
                                            feed_dict={itemFea_matrix: model.imageFeaMatrix,
                                                       userFea_matrix: model.R,
                                                       user_idx: i,
                                                       pos_item_idx: valid_pos_item,
                                                       neg_item_idx: valid_neg_item})
            final_acc += valid_acc
        print('Test Accuracy = %f' % (final_acc / model.nUsers))
        if final_acc/model.nUsers > max_acc:
            print('saving model. Accuracy = %f'%(final_acc/model.nUsers))
            saver.save(sess, './model/model_%f.ckpt' % (final_acc/model.nUsers), global_step=step)
            max_acc = final_acc/model.nUsers





