import numpy as np
import pandas as pd

def generate_random(data_size):
    random_dataset = {}
    random_dataset['sbp'] = np.random.uniform(100,220,data_size)
    random_dataset['tobacco'] = np.random.uniform(0,40,data_size)
    random_dataset['ldl'] = np.random.uniform(0,16,data_size)
    random_dataset['adiposity'] = np.random.uniform(0,50,data_size)
    random_dataset['famhist'] = np.random.randint(0,2,data_size)
    random_dataset['typea'] = np.random.randint(10,80,data_size)
    random_dataset['obesity'] = np.random.uniform(10,50,data_size)
    random_dataset['alcohol'] = np.random.uniform(0,170,data_size)
    random_dataset['age'] = np.random.randint(15,65,data_size)
    return pd.DataFrame(random_dataset,columns=random_dataset.keys())
