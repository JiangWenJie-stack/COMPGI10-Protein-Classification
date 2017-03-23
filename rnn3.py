# -*- coding: utf-8 -*-
"""
Code by Lewis Moffat

Structure of wrappers inspired by:
    https://danijar.com/structuring-your-tensorflow-models/
"""
import os
import functools
import tensorflow as tf
import numpy as np
import pdb
from Bio import SeqIO
import dictHot as dH
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import getData as gt
train = True


# Switches
batch_size = 256
opt_runs=500
vocab_size=21 
input_size=300
learning_rate_s=0.001     # 66, 0.001, 512, 256, 100, 
lstm_size=512   # size of hidden 'units'
stacked = False  # if true then the three stacked layers are added
lstmOr = False   # if true then LSTM cell is used instead of GRU
dropout = False # of true then add 50% droppout 
single=True    # if single than just use the dynamic rnn instead of bidirectional
attention=False # add attention mechanism
attn_len = 1
limit=500 # number of sequence to look at 



def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error

    @define_scope#(initializer=tf.contrib.slim.xavier_initializer(uniform=True, dtype=tf.float32))
    def prediction(self):
        ## this is where graph is defined
        x = self.image
        initz = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        # Word embeddings
        embeddings = tf.get_variable("W", [vocab_size, input_size], initializer=initz)
        x = tf.nn.embedding_lookup(embeddings, x) # [batch_size x max_seq_length x input_size]    
        
        # define cell
        if lstmOr==True:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=True, use_peepholes=False)  # create standard cell
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=lstm_size)
        if stacked==True:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 2, state_is_tuple=True) # add 3 layers
        if dropout == True:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)     # add dropout
        if attention==True:
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=attn_len, state_is_tuple=True)
        if single==True:
            seq_length=seqlen
            # run dynamic rnn so not having to worry about looping
            val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=seq_length)
            
            # get the last state
            
            val = tf.transpose(val,[1,0,2])
            val = tf.gather(val, int(val.get_shape()[0]) - 1)
            
            # perform the nonlinear layer and softmax layer 
            x = tf.contrib.layers.fully_connected(val, 256, activation_fn=None, biases_initializer=tf.zeros_initializer)
            #x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
            x = tf.nn.elu(x, 'elu')
        else:
            bs = tf.shape(x)[0]
            
            cellb=cell
            cellf=cell
            initial_statef = cellf.zero_state(bs, tf.float32)
            initial_stateb = cellb.zero_state(bs, tf.float32)
            
            
            seq_length=seqlen
            
            val, _ =tf.nn.bidirectional_dynamic_rnn(cellf, cellb, x, sequence_length=seq_length, initial_state_fw=initial_statef, initial_state_bw=initial_stateb )
            
            # get the last state
            val0 = tf.transpose(val[0],[1,0,2])
            val1 = tf.transpose(val[1],[1,0,2])
            val0 = tf.gather(val0, int(val0.get_shape()[0]) - 1)
            val1 = tf.gather(val1, int(val1.get_shape()[0]) - 1)
            val  = tf.concat(1,[val0,val1]) 
            
            # perform the nonlinear layer and softmax layer 
            x = tf.contrib.layers.fully_connected(val, 256, activation_fn=None, biases_initializer=tf.zeros_initializer)
            #x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
            x = tf.nn.elu(x, 'elu')
            
            # perform the nonlinear layer and softmax layer 
            x = tf.contrib.layers.fully_connected(val, 32, activation_fn=None, biases_initializer=tf.zeros_initializer)
            #x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
            x = tf.nn.elu(x, 'elu')
        
        #x = tf.contrib.layers.fully_connected(x, 20, activation_fn=None, biases_initializer=tf.zeros_initializer)
        #x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
        #x = tf.nn.relu(x, 'relu')
        
        x = tf.contrib.layers.fully_connected(x, 4, activation_fn=None, biases_initializer=tf.zeros_initializer)
        return x

    @define_scope
    def optimize(self):
        # Cost Function
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label)
        # Average of Cost
        avgCost=tf.reduce_mean(crossEntropy)
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = learning_rate_s
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.99, staircase=True)

        lambda_loss_amount = 0.000001
        Gradient_noise_scale = None
        Clip_gradients = 2.0
        # Loss, optimizer, evaluation
        l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if 'bias' not in tf_var.name)
        # Softmax loss and L2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label)) + l2
        # Gradient clipping Adam optimizer with gradient noise
        optimizr = tf.contrib.layers.optimize_loss(
            loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            clip_gradients=Clip_gradients,
            gradient_noise_scale=Gradient_noise_scale
        )
        
        
        
        return optimizr, avgCost

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def crossLoss(self):
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label)
        return tf.reduce_mean(crossEntropy)
    

def save_model(session):
    if not os.path.exists('models/model1/'):
        os.mkdir('models/model1/')
    saver = tf.train.Saver()
    saver.save(session, 'models/model1/model.ckpt')
    



"""
LSTM 128 1 layer 66.78%        #256 0.0001
LSTM 128 3 layer 66.09%
LSTM 256 1 Layer 68.69%
Bi-L 256 1 Layer 66.97%

Embeddings 
Bi-L 256 1 Layer 66.97%



"""



# reset potential graph
tf.reset_default_graph()
tf.Graph().as_default()



if train == True:
    with tf.Session() as sess:
        # place holder for tracking record of best score
        test=100
        # set up tf place holders
        image = tf.placeholder(tf.int32, [None, limit+30])
        label = tf.placeholder(tf.float32, [None, 4])
        phase = tf.placeholder(tf.bool, name='phase')
        seqlen = tf.placeholder(tf.int32, [None])
        # boot up instance of graph
        model = Model(image, label)
        sess.run(tf.global_variables_initializer())
        # lists for story data for later visualization
        trainE=[]
        testE=[]
        # convert training and test data to integer ID's 
        ta,tb,tc=gt.getData(t="test", setup=5, limit=limit)
        t1,t2,t3=gt.getData(t="train", setup=5, limit=limit)
        # for batch size decay
        #printcounter = 0
        for j in range(opt_runs):
            print('----- Epoch', j, '-----')
            images, seqLen, labels = ta,tb,tc
            error = sess.run(model.error, {image: images[:1153], label: labels[:1153], phase: 0, seqlen: seqLen[:1153]})
            error1 = sess.run(model.error, {image: images[1153:], label: labels[1153:], phase: 0, seqlen: seqLen[1153:]})
            error=(error+error1)/2
            print('Test Accuracy {:6.2f}%'.format(100 * (1-error)))
            testE.append(100 * (1-error))
            
            
            if error<test:
                test=error
                save_model(sess)
            
            images, sdl, labels = t1,t2,t3
            X, Y, Z = shuffle(images, labels, sdl, random_state=1)
            n = X.shape[0]
            
            #printcounter += 1
            #if printcounter == 10:
            #   batch_size=int(batch_size*0.99)
            #   printcounter=0
            
            
            for i in range(n // batch_size):
                images=X[i * batch_size: (i + 1) * batch_size]
                labels=Y[i * batch_size: (i + 1) * batch_size]
                seqLen=Z[i * batch_size: (i + 1) * batch_size]
                _, current_loss =sess.run(model.optimize, {image: images, label: labels, phase: 1, seqlen: seqLen})

            
        
        plt.figure()
        plt.plot(testE, label='Test Error')
        #plt.plot(trainE, label='Train Error')
        plt.title('Learning Curve of Final Model')
        plt.ylabel('% Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()
        
            
else:
    
    with tf.Session() as sess:
        # LOAD THE MODEL
        #data=dataPrep(False,"train")
        #testData=dataPrep(False,"test")
        sd, sdl, labels=gt.getData(t="train", setup=2)
        pdb.set_trace()
        image = tf.placeholder(tf.float32, [None, limit, 21])
        label = tf.placeholder(tf.float32, [None, 4])
        phase = tf.placeholder(tf.bool, name='phase')
        seqlen = tf.placeholder(tf.float32, [None])
        model = Model(image, label)
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        saver.restore(sess, 'models/model1/model.ckpt')
        

        images, seqLen, labels = sd, sdl, labels
        #images=np.concatenate(images[:])
        #images=images.reshape(-1,limit,21)
        #labels=np.concatenate(labels[:])
        #labels=labels.reshape(-1,4)
        error = sess.run(model.error, {image: images, label: labels, phase: 0, seqlen: seqLen})
        print('Test Accuracy {:6.2f}%'.format(100 * (1-error)))

        """
        images, labels = data[:,0], data[:,1]
        images=np.concatenate(images[:])
        images=images.reshape(-1,limit,21)
        labels=np.concatenate(labels[:])
        labels=labels.reshape(-1,4)
        error = sess.run(model.error, {image: images, label: labels, phase: 0})
        print('Train Accuracy {:6.2f}%'.format(100 * (1-error)))
        
        """