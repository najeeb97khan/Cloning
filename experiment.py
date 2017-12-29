'''
This script is a package that contains definitions
to run the experiment for testing pure random data
on the black box and white box
'''

## Importing standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

## Importing utils
from utils import generate_pure_random

## Class to handle random data and models

class Random(object):
    
    def __init__(self, X, y, feature_names):
        
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.models = {}
        self.X_random = None
        self.y_random = {}
        
    def create_random_data(self, generator, data_size, data_needed=False):
        
        if data_needed:
            self.X_random = generator(self.X, self.feature_names, data_size).values
        else:
            self.X_random = generator(self.feature_names, data_size).values
    
    def fit_blackbox_models(self, models, names):
        
        score, i = [], 0
        for model in models:
            model.fit(self.X, self.y)
            score.append(np.mean(np.asarray(cross_val_score(model, self.X, self.y))))
            self.models[names[i]] = model
            i += 1
        return score
    
    def generate_labels(self, names):
        
        i = 0
        for name, model in self.models.iteritems():
            y_random = model.predict(self.X_random)
            self.y_random[names[i]] = y_random
            print 'Model : ',model
            print 'Distribution: ', Counter(y_random)
            i += 1
            
    def report_whitebox_accuracy(self, whitebox, names):
        i = 0
        accuracy = {}
        for model in whitebox:
            scores = []
            for name, y_random in self.y_random.iteritems():
                
                try:
                    score = 0
                    for k in range(5):
                        
                        model.fit(self.X_random, y_random)
                        score += model.score(self.X, self.y)
                    
                    score = score/5
                        
                except:
                    score=np.nan
                scores.append(score)
                print 'Blackbox: ',name, ' Whitebox: ', names[i], ' Accuracy: ', score
            accuracy[names[i]] = scores
            i += 1
        return accuracy
    
    def plot(self, whitebox_acc, names):
        pd.DataFrame(whitebox_acc,columns=names,index=whitebox_acc.keys()).plot(kind='bar')
        plt.xlabel('White box models')
        plt.ylabel('Accuracy')
        plt.title('Whitebox and Blackbox Comparision')