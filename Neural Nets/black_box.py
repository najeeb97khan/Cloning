import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import sys


sys.path.insert(0,'../utils')
from generate_random import generate_random

class blackBox(object):

    def __init__(self,params,layer_name):
        with tf.variable_scope(layer_name) as scope:
            self.batch_size = params['batch_size']
            self.learning_rate = params['learning_rate']
            self.n_epochs = params['n_epochs']
            self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")
            self.placeholders = None
            self.loss = None
            self.optimizer = None
            self.summary_op = None
            self.ckpt = None
            self.accuracy = None
            self.writer = None
            self.output = None
            self.saver = None

    def _create_placeholder(self,num_input,num_output,layer_name):
        with tf.variable_scope(layer_name) as scope:
            image = tf.placeholder(tf.float32,shape=[None,num_input],name="Input")
            label = tf.placeholder(tf.float32,shape=[None,num_output],name="Label")
            self.placeholder = (image,label)
            return (image,label)

    def _create_hidden_layer(self,prev_layer,num_hidden,layer_name):
        with tf.variable_scope(layer_name) as scope:
            fan_in = prev_layer.get_shape().as_list()[1]
            w = tf.get_variable(name="weights",shape=[fan_in,num_hidden],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="biases",shape=[num_hidden],initializer=tf.random_normal_initializer())
            act = tf.matmul(prev_layer,w) + b
            out = tf.nn.relu(act)
            return out

    def _create_softmax(self,prev_layer,num_output,layer_name):
        with tf.variable_scope(layer_name) as scope:
            fan_in = prev_layer.get_shape().as_list()[1]
            w = tf.get_variable(name="weights",shape=[fan_in,num_output],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="biases",shape=[num_output],initializer=tf.random_normal_initializer())
            act = tf.matmul(prev_layer,w) + b
            out = tf.nn.softmax(act)
            self.output = out
            return out

    def _create_loss(self,correct_preds,output_preds,layer_name):
        with tf.variable_scope(layer_name) as scope:
            loss = tf.reduce_mean(-tf.reduce_sum(correct_preds*tf.log(output_preds),reduction_indices=[1]))
            self.loss = loss

    def _create_optimizer(self,layer_name):
        with tf.variable_scope(layer_name) as scope:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)       
    
    def _create_summaries(self,layer_name):
        with tf.variable_scope(layer_name) as scope:
            tf.summary.scalar("loss",self.loss)
            tf.summary.histogram("loss",self.loss)
            ## Uncomment following lines to turn of summaries
            ## self.summary_op = tf.constant(1)
            ## return
            self.summary_op = tf.summary.merge_all()

    def _accuracy(self,correct_preds,output_preds,layer_name):
        with tf.variable_scope(layer_name) as scope:
            correct_predictions = tf.equal(tf.arg_max(output_preds,1),tf.arg_max(correct_preds,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
    
    def predict(self,test_data,file_name,test_labels=None):
        self.saver = tf.train.Saver()
        image,label = self.placeholder
        with tf.Session() as sess:
            self.ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/{}/checkpoint'.format(file_name)))
            if self.ckpt and self.ckpt.model_checkpoint_path:
                print 'Restoring checkpoint...'
                self.saver.restore(sess,self.ckpt.model_checkpoint_path)
            if test_labels == None:

                output_pred = sess.run([self.output],feed_dict={image:test_data})
                output_pred = np.asarray(output_pred)
                output_pred = np.asarray(output_pred).reshape(output_pred.shape[1],output_pred.shape[2])
                return output_pred

            else:

                 output_pred,acc = sess.run([self.output,self.accuracy],feed_dict={image:test_data,label:test_labels})
                 print np.asarray(output_pred).shape
                 print 'Test Accuracy: {}'.format(acc)
                 output_pred = np.asarray(output_pred).reshape(output_pred.shape[0],output_pred.shape[1])
                 return np.argmax(output_pred,1)


    def train(self,data,file_name):
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            self.writer = tf.summary.FileWriter('../graphs/{}'.format(file_name),sess.graph)
            self.ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/{}/checkpoint'.format(file_name)))
            if self.ckpt and self.ckpt.model_checkpoint_path:
                print 'Restoring checkpoint...'
                self.saver.restore(sess,self.ckpt.model_checkpoint_path)
            
            try:
                n_batches = data.train.images.shape[0]/self.batch_size
            except:
                n_batches = data[0].shape[0]/self.batch_size

            image,y = self.placeholder
            for i in range(1,self.n_epochs+1):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                for j in tqdm(range(n_batches)):
                    try:
                        x_batch,y_batch = data.train.next_batch(self.batch_size)
                    except:
                        cur_batch = j*self.batch_size
                        nex_batch = (j+1)*self.batch_size
                        x_batch,y_batch = data[0][cur_batch:nex_batch],data[1][cur_batch:nex_batch]
    
                    _,l,acc,summary = sess.run([self.optimizer,self.loss,self.accuracy,self.summary_op],
                                                feed_dict={image:x_batch,y:y_batch})
                    epoch_loss += l
                    epoch_accuracy += acc
                
                ## Comment the following line to turn off summaries
                self.writer.add_summary(summary,global_step=i)
                if i % 5 == 0:
                    print 'Saving Checkpoint...'
                    self.saver.save(sess,'../checkpoints/{}/{}'.format(file_name,file_name))
                
                print 'Epoch: {}\tLoss: {}\tAccuracy: {}'.format(i,epoch_loss,epoch_accuracy/n_batches)

def main():
    
    ## Reading in the data
    data = pd.read_csv('../data/heart.csv')
    labels = pd.get_dummies(data.pop('chd'))
    data['famhist'] = data['famhist'].map({"Absent":0,"Present":1})
    input_ = data.values
    print input_.shape
    print labels.shape
    
    ## Defining the parameters
    params = {}
    params['batch_size'] = 100
    params['learning_rate'] = 1e-4
    params['n_epochs'] = 10
    NUM_FEATURES = 9
    NUM_CLASSES = 2
    NUM_NEURONES = 1024
    
    ## Assembling the graph
    print 'Assembling the graph....'
    model = blackBox(params,"black_box")
    image,label = model._create_placeholder(num_input=NUM_FEATURES,num_output=NUM_CLASSES,layer_name="placeholder_bb")
    hidden_1 = model._create_hidden_layer(image,NUM_NEURONES,"hidden_1_bb")
    hidden_2 = model._create_hidden_layer(hidden_1,NUM_NEURONES,"hidden_2_bb")
    softmax = model._create_softmax(hidden_2,NUM_CLASSES,"softmax_bb")
    model._create_loss(label,softmax,"loss_bb")
    model._create_optimizer("optimizer_bb")
    model._accuracy(label,softmax,"accuracy_bb")
    model._create_summaries("summary_bb")
    
    ## Training the graph
    if str(raw_input('Want to train the model: ')).lower() == 'y':    
        print 'Training the model...'
        model.train((input_,labels),file_name="black_box")

    ## Generating random data
    data = generate_random(100000)
    print data.shape
    with open("../data/data.pkl","wb") as fp:
        pkl.dump(data,fp)

    ## Output for white box is same as output from black box for given data
    output = model.predict(data,file_name="black_box")
    print output.shape

    ## Storing output to a .csv file
    with open("../data/label.pkl","wb") as fp:
        pkl.dump(output,fp)

if __name__ == "__main__":
    main()