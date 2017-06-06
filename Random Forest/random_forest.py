## Appllying black box cloning technique to Random Forest Classifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.examples.tutorials.mnist import input_data
import sys

sys.path.insert(0,'../utils')
from generate_cifar import CIFAR

## Getting in the data
print 'Random Forest Classifier ==== Heart Dataset'
df_original = pd.read_csv('../data/heart.csv')
label = df_original.pop('chd')
df_original['famhist'] = df_original['famhist'].map({'Absent':0,'Present':1})
data = df_original.values
print 'Heart Dataset Characteristic'
print data.shape,label.shape

## Getting in the random dataset
import pickle as pkl
with open("../data/data.pkl","rb") as fp:
    df_random = pkl.load(fp)

## Training the Random Forest
print 'Creating the black box'
rfc = RandomForestClassifier()
rfc.fit(data,label)

## Generating labels for random dataset
print 'Generating labels for random samples using black box'
label_random = rfc.predict(df_random)
print label_random.shape

## Training white box 
print 'Training white box on random samples'
model = RandomForestClassifier()
model.fit(df_random,label_random)

print 'Performance of Black Box on actual dataset'
print 'Accuracy: {}'.format(rfc.score(data,label)*100)
print 'Performance of White Box on actual dataset'
print 'Accuracy: {}'.format(model.score(data,label)*100)


## MNIST DATASET
## Getting the data
print 'Random Forest Classifier ==== Image Dataset'
mnist = input_data.read_data_sets('../data/MNIST_data/',one_hot=True)
print 'Characteristic of Image Dataset'
print mnist.train.images.shape, mnist.train.labels.shape

## Training the black box
print 'Creating the black box'
rfc_black = RandomForestClassifier()
rfc_black.fit(mnist.train.images,np.argmax(mnist.train.labels,axis=1))

## Generating labels for Random Dataset
random_dataset = np.random.randint(0,256,(10000,784))
label_random = rfc_black.predict(random_dataset)
print label_random.shape

## Training the white box
print 'Training white box on random images'
rfc_white = RandomForestClassifier()
rfc_white.fit(random_dataset,label_random)

print 'Performance of Black Box on actual dataset'
print 'Accuracy: {}'.format(rfc_black.score(mnist.train.images,np.argmax(mnist.train.labels,1))*100)
print 'Performance of White Box on actual dataset'
print 'Accuracy: {}'.format(rfc_white.score(mnist.train.images,np.argmax(mnist.train.labels,1))*100)

## Random dataset as random images
print 'Random dataset as random set of images (CIFAR-10)'
path = '../../../Dataset/CIFAR-10/*'
cifar = CIFAR(path)
print cifar
new_data = cifar.resize_image((28,28,3))
print new_data.shape
random_dataset = cifar.drop_color_channel(new_data,(28,28,3),(28,28))
print random_dataset.shape

print 'Generating labels from black box for CIFAR-10'
label_random = rfc_black.predict(random_dataset)
print label_random.shape

print 'Training white box on random images (CIFAR-10)'
rfc_white = RandomForestClassifier()
rfc_white.fit(random_dataset,label_random)

print 'Performance of White Box on actual dataset after trained on CIFAR-10'
print 'Accuracy: {}'.format(rfc_white.score(mnist.train.images,np.argmax(mnist.train.labels,1))*100)
