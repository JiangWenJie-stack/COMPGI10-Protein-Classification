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

train = True


# Switches
batch_size = 256
opt_runs=300
learning_rate_s=0.001
lstm_size=128   # size of hidden 'units'
stacked = False  # if true then the three stacked layers are added
lstmOr = False   # if true then LSTM cell is used instead of GRU
dropout = False # of true then add 50% droppout 
single=False     # if single than just use the dynamic rnn instead of bidirectional
attention=False # add attention mechanism
attn_len = 2


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
        
        #x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
        
        # define cell
        if lstmOr==True:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=True, use_peepholes=False)  # create standard cell
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=lstm_size)
        if stacked==True:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 3, state_is_tuple=True) # add 3 layers
        if dropout == True:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)     # add dropout
        if attention==True:
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=attn_len, state_is_tuple=True)
        if single==True:
            # run dynamic rnn so not having to worry about looping
            val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            
            # get the last state
            val = tf.transpose(val,[1,0,2])
            val = tf.gather(val, int(val.get_shape()[0]) - 1)
        else:
            bs = tf.shape(x)[0]
            
            cellb=cell
            cellf=cell
            initial_statef = cellf.zero_state(bs, tf.float32)
            initial_stateb = cellb.zero_state(bs, tf.float32)
            
            
            seq_length=tf.fill([bs],limit)
            
            val, _ =tf.nn.bidirectional_dynamic_rnn(cellf, cellb, x, sequence_length=seq_length, initial_state_fw=initial_statef, initial_state_bw=initial_stateb )
            
            # get the last state
            val0 = tf.transpose(val[0],[1,0,2])
            val1 = tf.transpose(val[1],[1,0,2])
            #
            val0 = tf.gather(val0, int(val0.get_shape()[0]) - 1)
            val1 = tf.gather(val1, int(val1.get_shape()[0]) - 1)
            val  = tf.concat(1,[val0,val1]) 
            
        # perform the nonlinear layer and softmax layer 
        x = tf.contrib.layers.fully_connected(val, 128, activation_fn=None, biases_initializer=tf.zeros_initializer)
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
    if not os.path.exists('models/model1d/'):
        os.mkdir('models/model1d/')
    saver = tf.train.Saver()
    saver.save(session, 'models/model1d/model.ckpt')
    

def dataPrep(trains=True,extension="train",limit=200):
    # read the fasta files
    if trains==True:
        mito = list(SeqIO.parse(extension+"/mito.fasta", "fasta"))
        secreted = list(SeqIO.parse(extension+"/secreted.fasta", "fasta"))
        nucleus = list(SeqIO.parse(extension+"/nucleus.fasta", "fasta"))
        cyto = list(SeqIO.parse(extension+"/cyto.fasta", "fasta"))
        
        data=[mito,secreted,nucleus,cyto]
        """
        mito      = [1,0,0,0]
        secreted  = [0,1,0,0]
        nucleus   = [0,0,1,0]
        cyto      = [0,0,0,1]
        """
        pData=[]
        
        for idx, dat in enumerate(data):
            for seq in dat:
                if len(str(seq.seq))>=limit:
                    conv=str(seq.seq)
                    conv=conv[::-1]
                    newS=conv[:int(limit/2)]+conv[-int(limit/2):]
                    newS=list(newS)
                    finalS=[]
                    for char in newS:
                        finalS.append(dH.oneHot(char))
                    newS=np.array(finalS)
                else:
                    conv=str(seq.seq)
                    conv=conv[::-1]
                    pad=[0] * (limit-len(conv))
                    newS=list(conv[:len(conv)//2])+pad+list(conv[len(conv)//2:])
                    finalS=[]
                    for char in newS:
                        finalS.append(dH.oneHot(char))
                    newS=np.array(finalS)
                if len(newS)!=limit:
                    print("Something stoped working")
                    
                label = np.zeros((4,1))
                label[idx,0]=1
                pData.append([newS,label])
            
            print("Done with {}".format(idx))
        np.save(extension+"/onehot2",pData)
        return np.load(extension+"/onehot2.npy")
        
    else:
        return np.load(extension+"/onehot2.npy")

"""
LSTM 128 1 layer 66.78%        #256 0.0001
LSTM 128 3 layer 66.09%
LSTM 256 1 Layer 68.69%
Bi-L 256 1 Layer 66.97%
"""



# reset potential graph
tf.reset_default_graph()
tf.Graph().as_default()
test=100
limit=1000
#64.8 - 100        256 b 256 size 0.001 all false, grad=2.0
#67.4 - 200
#67.4 - 200        256 b 256 size 0.001 all false, grad=false  
#67.9 - 250        256 b 256 size 0.001 all false, grad=2.0 
#68.8 - 300        256 b 256 size 0.001 all false, grad=2.0 
#70.50 - 1000

if train == True:
    with tf.Session() as sess:

        data=dataPrep(True,"train",limit)
        testData=dataPrep(True,"test",limit)
        image = tf.placeholder(tf.float32, [None, limit, 21])
        label = tf.placeholder(tf.float32, [None, 4])
        phase = tf.placeholder(tf.bool, name='phase')
        model = Model(image, label)
        sess.run(tf.global_variables_initializer())
        n = data.shape[0]
        
        trainE=[]
        testE=[]
        
        for j in range(opt_runs):
            print('----- Epoch', j, '-----')
            images, labels = testData[:,0], testData[:,1]
            images=np.concatenate(images[:])
            
            images=images.reshape(-1,limit,21)
            labels=np.concatenate(labels[:])
            labels=labels.reshape(-1,4)
            error1 = sess.run(model.error, {image: images[:500], label: labels[:500], phase: 0})
            error2 = sess.run(model.error, {image: images[500:1000], label: labels[500:1000], phase: 0})
            error3 = sess.run(model.error, {image: images[1000:1500], label: labels[1000:1500], phase: 0})
            error4 = sess.run(model.error, {image: images[1500:2000], label: labels[1500:2000], phase: 0})
            error5 = sess.run(model.error, {image: images[2000:], label: labels[2000:], phase: 0})
            error=(error1+error2+error3+error4+error5)/5
            print('Test Accuracy {:6.2f}%'.format(100 * (1-error)))
            
            testE.append(100 * (1-error))
            
            total_loss = 0
            
            if error<test:
                test=error
                save_model(sess)
            
            images, labels = data[:,0], data[:,1]
            images=np.concatenate(images[:])
            images=images.reshape(-1,limit,21)
            labels=np.concatenate(labels[:])
            labels=labels.reshape(-1,4)
            """
            error = sess.run(model.error, {image: images, label: labels, phase: 0})
            print('Train Accuracy {:6.2f}%'.format(100 * (1-error)))
            
            trainE.append(100 * (1-error))
            """
            X, Y = shuffle(images, labels)
            
            for i in range(n // batch_size):
                images=X[i * batch_size: (i + 1) * batch_size]
                labels=Y[i * batch_size: (i + 1) * batch_size]
                _, current_loss =sess.run(model.optimize, {image: images, label: labels, phase: 1})
                total_loss += current_loss

            
        
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
        data=dataPrep(True,"train",limit)
        testData=dataPrep(True,"test",limit)
        image = tf.placeholder(tf.float32, [None, limit, 21])
        label = tf.placeholder(tf.float32, [None, 4])
        phase = tf.placeholder(tf.bool, name='phase')
        model = Model(image, label)
        sess.run(tf.global_variables_initializer())
        n = data.shape[0]
        
        trainE=[]
        testE=[]
        
        saver = tf.train.Saver()
        saver.restore(sess, 'models/model1b/model.ckpt')
        

        images, labels = testData[:,0], testData[:,1]
        images=np.concatenate(images[:])
        
        images=images.reshape(-1,limit,21)
        labels=np.concatenate(labels[:])
        labels=labels.reshape(-1,4)
        error1 = sess.run(model.error, {image: images[:500], label: labels[:500], phase: 0})
        error2 = sess.run(model.error, {image: images[500:1000], label: labels[500:1000], phase: 0})
        error3 = sess.run(model.error, {image: images[1000:1500], label: labels[1000:1500], phase: 0})
        error4 = sess.run(model.error, {image: images[1500:2000], label: labels[1500:2000], phase: 0})
        error5 = sess.run(model.error, {image: images[2000:], label: labels[2000:], phase: 0})
        error=(error1+error2+error3+error4+error5)/5
        print('Test Accuracy {:6.2f}%'.format(100 * (1-error)))
        
        testE.append(100 * (1-error))
        """
        images, labels = data[:,0], data[:,1]
        images=np.concatenate(images[:])
        images=images.reshape(-1,limit,21)
        labels=np.concatenate(labels[:])
        labels=labels.reshape(-1,4)
        error = sess.run(model.error, {image: images, label: labels, phase: 0})
        print('Train Accuracy {:6.2f}%'.format(100 * (1-error)))
        
        """