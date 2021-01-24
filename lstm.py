import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from data_reader import load_data
import numpy as np
from uncompress import *
import os



def one_hot(batch_size,Y):

    B = np.zeros((batch_size,2))

    B[np.arange(batch_size),Y] = 1

    return B


if __name__=='__main__':

    
    # print one_hot(3,np.array((1,0,1)))
    # exit(0)
    # Training Parameters
    learning_rate = 0.001
    num_epoch = 4
    batch_size = 2
    display_step = 1
    input_size = 50
    num_classes = 2 
    lstm_size = 256

    X = tf.placeholder(tf.float32, [None, input_size, 86796])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    #logits = conv_net(X)

    #cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # # print cell.output_shape
    #stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell]*2)

    #print tf.shape(stacked_lstm)
    # exit(0)

    # create 2 LSTMCells
    rnn_layers = [tf.keras.layers.LSTMCell(size) for size in [256, 256]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.keras.layers.StackedRNNCells(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    output, state = tf.keras.layers.RNN(cell=multi_rnn_cell,
                                    inputs=None,
                                    dtype=tf.float32)
  
    
    #data = tf.placeholder(tf.float32, [None, input_size, 86796])
    #output, state = tf.nn.dynamic_rnn(stacked_lstm, X, dtype=tf.float32)
    
    # Select last output.
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
        
    slim = tf.contrib.slim
    logits = slim.fully_connected(last, 2, scope='fc',activation_fn=None,weights_initializer=tf.truncated_normal_initializer(0.0, 0.01))


    prediction = tf.nn.softmax(logits)
    # # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
         logits=logits, labels=Y))
    tf.summary.scalar('loss',loss_op)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)


    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

   