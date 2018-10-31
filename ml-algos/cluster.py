import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from copy import copy
import random
import os


np.set_printoptions(suppress=True)
"""this is fulfilled as a function in the cluster_class.py script"""
data = pd.read_csv('/Users/bennicholl/Desktop/reengineereddb2.csv')

#"""add one hot encoded labels. might want to save this to a csv after labels are added"""
data['labels'] = new_y
        
    
"""creates binary labels. EX for SECOND combionation:  label vecotr [0 1 0 0] == 1,   label vecotr [0 0 0 1] == 0"""
def create_binary_training_ex(db, label_names):
    label_names = copy(label_names)
    individual_dataframes = []
    """iterate through each combo of feature names. EX of 1 combo: ['running_status', 'drive_voltage_monitor', 'bus_voltage']"""
    for index_of_combo,combo in enumerate(label_names):
        """create binary labels. EX for SECOND combionation:  label vecotr [0 1 0 0] == 1,   label vecotr [0 0 0 1] == 0"""
        binary_labels = []
        """ if we have two combos twice, such as: ['running_status', 'tubing_prs'], ['running_status', 'tubing_prs']. we only 
        want to add 'labels' to this combination one"""
        if 'labels' not in combo:           
            combo.append('labels')
            
        """get the dataframe associated with the above combination"""
        dataframe_of_combos = db[combo]
        for e,training_vector in dataframe_of_combos.iterrows():
            if training_vector['labels'][index_of_combo] == 1:
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        """append the binary labels as a column to the dataframe"""
        dataframe_of_combos['binary_labels'] = binary_labels
        """drop the one hot encoded multivariate labels column"""
        dataframe_of_combos = dataframe_of_combos.drop('labels',1)
        individual_dataframes.append(dataframe_of_combos)
        
    return individual_dataframes
        
# binary_training_examples = create_binary_training_ex(data, b.filtered_probs_label_names)
