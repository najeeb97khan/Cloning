import tensorflow as tf
import numpy as np
from black_box import blackBox
from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd

def main():
    
    tf.reset_default_graph()
    
    ## Loading the data
    print "Loading the data..."
    with open("../data/data.pkl","rb") as fp:
        data = pkl.load(fp)

    with open("../data/label.pkl","rb") as fp:
        output = pkl.load(fp)
    print data.shape
    print output.shape

    ## Defining the parameters
    NUM_CLASSES = 2
    NUM_FEATURES = 9
    NUM_NEURONES = 1024
    params = {}
    params['batch_size'] = 100
    params['learning_rate'] = 1e-5
    params['n_epochs'] = 10
    
    ## White Box Initialisation
    print "Creating the graph"
    model_wb = blackBox(params,"white_box")
    image_wb,label_wb = model_wb._create_placeholder(num_input=NUM_FEATURES,num_output=NUM_CLASSES,layer_name="placeholder_wb")
    hidden_1_wb = model_wb._create_hidden_layer(image_wb,NUM_NEURONES,"hidden_1_wb")
    hidden_2_wb = model_wb._create_hidden_layer(hidden_1_wb,NUM_NEURONES,"hidden_2_wb")
    softmax_wb = model_wb._create_softmax(hidden_2_wb,NUM_CLASSES,"softmax_wb")
    model_wb._create_loss(label_wb,softmax_wb,"loss_wb")
    model_wb._create_optimizer("optimizer_wb")
    model_wb._accuracy(label_wb,softmax_wb,"accuracy_wb")
    model_wb._create_summaries("summary_wb")

    ## Getting the original dataset
    df = pd.read_csv('../data/heart.csv')
    labels = df.pop('chd')
    df['famhist'] = df['famhist'].map({"Absent":0,"Present":1})
    input_ = df.values
    print input_.shape
    print labels.shape

    ## Training the model
    ch = str(raw_input('Want to train the model?: '))
    if ch == 'y' or ch == 'Y':
        
        print "Training the model..."
        dataset = (data,output)
        model_wb.train(dataset,"white_box")
    
    ## Prediciting on original dataset
    predictions = model_wb.predict(input_,file_name="white_box")
    predictions = np.argmax(predictions,axis=1)
    print 'Accuracy: ',np.mean(np.equal(predictions,labels))

if __name__ == "__main__":
    main()