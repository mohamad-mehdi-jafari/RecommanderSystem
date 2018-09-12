'''
    create by mehdi jafari 
    9 sep 2018
'''

# configuration panel 
train_set_path = 'train_user_item_score.txt'
validation_set_path = 'validation_user_item_score.txt'
users_connection_path = 'users_connections.txt'
connection_value = 0
batch_size = 16
hidden_dimenssion = 10
users_count = 17616 # train set 
items_count = 16122 # train set 

import re
import numpy as np 
import tensorflow as tf


def generate_constraint():
    with open(train_set_path, 'r') as file:
        train = file.read()
    train = [int(s) for s in re.findall(r'\d+', train)]
    with open(users_connection_path, 'r') as file:
        connection = file.read()
    connection = [int(s) for s in re.findall(r'\d+', connection)]
    with open(validation_set_path, 'r') as file:
        validation = file.read()
    validation = [int(s) for s in re.findall(r'\d+', validation)]
    while(True):
        for user in range(1, max(train[0::3])):
            for row in range(0, len(train)//3):
                if(user==train[row*3]):
                    yield [user]+train[row*3+1:row*3+3]
            
    
def generate_batch():
    sample = generate_constraint()
    batch = []
    while(True):
        for i in range(0, batch_size):
            batch.append(next(sample))
        yield batch
        batch = []
        
batch = generate_batch()

def gen():
    counter = 0 
    while True:
        my_batch = next(batch)
        my_batch = np.array(my_batch, dtype=np.int16)
        vector_ID1 = my_batch[:, 0:1]
        vector_ID1 = np.reshape(vector_ID1, newshape=(batch_size))
        vector_ID2 = my_batch[:, 1:2]
        vector_ID2 = np.reshape(vector_ID2, newshape=(batch_size))
        target_values = my_batch[:, 2:3]
        target_values = np.reshape(target_values, newshape=(batch_size))
        counter += 1
#        print("this is it")
#        print(vector_ID1, vector_ID2, target_values)
        yield target_values, vector_ID1, vector_ID2
        

    
        

def graph():
    
    vector_ID1 = tf.placeholder(dtype=tf.int32, shape=(None,))
    vector_ID2 = tf.placeholder(dtype=tf.int32, shape=(None,))
    target_value = tf.placeholder(dtype=tf.float32, shape=(None,))
    

#    # define weights.
#    # In our recommander system, it's the weights that we care about
    users_theta_matrix = tf.get_variable('users_theta',
                                   shape=[users_count, hidden_dimenssion],
                                   initializer=tf.random_uniform_initializer())
    
    
    items_theta_matrix = tf.get_variable('items_theta',
                                   shape=[items_count, hidden_dimenssion],
                                   initializer=tf.random_uniform_initializer())
#
#    # define lookup table using weight matrix and our indecis
    users = tf.nn.embedding_lookup(users_theta_matrix, 
                                   vector_ID1, name='users')
    items = tf.nn.embedding_lookup(items_theta_matrix, 
                                   vector_ID2, name='items')
    
    init = tf.global_variables_initializer()
    mul = tf.diag_part(tf.matmul(users, tf.transpose(items)))
    loss = tf.reduce_mean(tf.square(mul-target_value))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    saver = tf.train.Saver()
    return (vector_ID1, vector_ID2, target_value, init, loss, train_op, saver)
    
    
#

b = gen()


if __name__ == "__main__":
#    dataset = tf.data.Dataset.from_generator(gen,
#                    (tf.int32, tf.int32, tf.int32),
#                    (tf.TensorShape([batch_size]), 
#                     tf.TensorShape([batch_size]),
#                     tf.TensorShape([batch_size])))
    
    vector_1, vector_2, sim, init, loss, train_op, saver = graph()
    with tf.Session() as sess:
        tgt, v_id1, v_id2 = next(b)
        sess.run([init], feed_dict = {vector_1:v_id1, 
                                     vector_2:v_id2, sim:tgt})
        
        
        
        for step in range(0, 1000000):
            
            sess.run([train_op], feed_dict = {vector_1:v_id1, 
                                             vector_2:v_id2, sim:tgt})
            if step%10000==0:
                print(sess.run([loss], feed_dict = {vector_1:v_id1, 
                                             vector_2:v_id2, sim:tgt}))
            tgt, v_id1, v_id2 = next(b)
            if step%10000==0:
                save_path = saver.save(sess, "trained_NN/RS{}.ckpt".format(step))
                print("models is saved in ", save_path)
#            print(sess.run([loss], feed_dict = {v1:v_id1, v2:v_id2, t:tgt}))
##        print(sess.run([u], feed_dict = {v1:v_id1, v2:v_id2, t:tgt}))
#        print(sess.run([t], feed_dict = {v1:v_id1, v2:v_id2, t:tgt}))
#



