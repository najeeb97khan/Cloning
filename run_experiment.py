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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle as pkl

## Importing utils
from utils import generate_pure_random, generate_constrained_random

## Imporitng random package
from experiment import Random

## Silence warnings
import warnings
warnings.filterwarnings('ignore')

def perform_analysis(X, y, feature_names):
    
    random = Random(X, y,feature_names)
    random.create_random_data(generate_constrained_random, 1000, data_needed=True)
    #models = [RandomForestClassifier()]
    models = [RandomForestClassifier(), LogisticRegression(), KNeighborsClassifier()]
    #names = ['RFC']
    names = ['RFC', 'LR', 'KNN']
    blacbox_acc = random.fit_blackbox_models(models, names)
    
    for i in range(len(names)):
        print names[i], ' : ', blacbox_acc[i]

    print "*"*50
    
    random.generate_labels(names)
    whitebox = models
    whitebox_acc = random.report_whitebox_accuracy(whitebox, names)


def main():
    
    print '1. Heart Dataset'
    print '2. Breast Cancer Dataset'
    print '3. Adult Census Dataset'
    print '4. Bridges Dataset'
    print '5. Mushrooms Dataset'

    i = 0
    ch = int(raw_input('Enter your choice: '))

    print 

    if ch == 1:
    
        while(i < 5):
            
            # Heart Dataset
            print '========================== Heart Dataset ============================'
            data = pd.read_csv('../data/heart.csv')
            data['famhist'] = data['famhist'].map({"Absent": 0, "Present":1})
            y = data.pop("chd").values
            X = data.values
            feature_names = data.columns
            print X.shape, y.shape
            perform_analysis(X, y, feature_names)
            i += 1

    if ch == 2:
        
        while(i < 5):        
        
            # Breast Cancer Dataset
            print '========================== Breast Cancer Dataset =========================='
            data = datasets.load_breast_cancer()
            X = data.data
            y = data.target
            feature_names = data.feature_names
            print X.shape, y.shape
            perform_analysis(X, y, feature_names)
            i += 1
    
    if ch == 3:

        while(i < 5):
                
            ## Adult Census dataset
            print '========================== Adult Census Dataset ==========================='
            with open('../../../Dataset/Census/train.pkl', 'rb') as fp:
                data = pkl.load(fp)
            with open('../../../Dataset/Census/label.pkl', 'rb') as fp:
                label = pkl.load(fp)
            feature_names = data.columns
            X = data.values
            y = label
            print X.shape, y.shape
            perform_analysis(X, y, feature_names)
            i += 1

    if ch == 4:
        
        while(i < 5):
            
    
            ## Bridges Dataset
            print '========================= Bridges Dataset ============================='
            with open('../../../Dataset/Bridges/train.pkl', 'rb') as fp:
                data = pkl.load(fp)
            with open('../../../Dataset/Bridges/label.pkl', 'rb') as fp:
                label = pkl.load(fp)
            feature_names = data.columns
            X = data.values
            y = label
            print X.shape, y.shape
            perform_analysis(X, y, feature_names)
            i += 1
    
    if ch == 5:
        
        while(i < 5):
                

            ## Mushroom Datasey
            print '======================== Mushroom Dataset ==========================='
            with open('../../../Dataset/Mushroom/train.pkl', 'rb') as fp:
                data = pkl.load(fp)
            with open('../../../Dataset/Mushroom/label.pkl', 'rb') as fp:
                label = pkl.load(fp)
            feature_names = data.columns
            X = data.values
            y = label
            print X.shape, y.shape
            perform_analysis(X, y, feature_names)
            i += 1



    

if __name__ == "__main__":
    main()
