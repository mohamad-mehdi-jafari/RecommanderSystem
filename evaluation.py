#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:11:31 2018

@author: meti
"""

# configuration panel 
validation_set_path = 'validation_user_item_score.txt'
hidden_dimenssion = 10
users_count = 17616 # train set 
items_count = 16122 # train set 



import tensorflow as tf
import numpy as np 
# rastore our model variables first 
users_theta_matrix = tf.get_variable('users_theta',
                                   shape=[users_count, hidden_dimenssion],
                                   initializer=tf.zeros_initializer())
    
    
items_theta_matrix = tf.get_variable('items_theta',
                               shape=[items_count, hidden_dimenssion],
                               initializer=tf.zeros_initializer())
users_matrix = None
items_matrix = None
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "trained_NN/RS170000.ckpt")
    users_matrix = sess.run(users_theta_matrix)
    items_matrix = sess.run(items_theta_matrix)
###########################################################
    


def predict(users, items, user_id, item_id):
    return np.inner(users[user_id, :], items[item_id, :])

with open(validation_set_path, 'r') as file:
    validation = file.read()
    validation = [int(s) for s in re.findall(r'\d+', validation)]

predictions = []
users_id = validation[0::3]
items_id = validation[1::3]
scores = validation[2::3]
for i in range(0, len(scores)):
    predictions.append(predict(users_matrix, items_matrix, 
                               users_id[i], items_id[i]))

print(np.sqrt(np.mean((np.array(predictions)-np.array(scores))**2)))
