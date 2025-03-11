
from tensorflow import keras
import numpy as np
import os
import random 
import tensorflow as tf
import math
class Load_data_RNA(keras.utils.Sequence):

    """generate data in sequence mode for training the neural network"""

    def __init__(self, batch_size, path, files_list, labels):
        self.batch_size = batch_size
        self.X_train = np.zeros([self.batch_size,400,1]) #Window size is 400 measurements
        self.X_train2 = np.zeros([self.batch_size,40,4,1]) #Maximal 40 bases within window of 400 measurements assumed
        self.labels = np.zeros([self.batch_size,40,labels])

        # The set of characters accepted in the transcription.
        self.path = path
        self.files_list = files_list
        self.ind_rand = np.arange(0,len(self.files_list),1)
        np.random.shuffle(self.ind_rand)
        np.random.shuffle(self.files_list)
        self.instances = 0
        for batch_index,batch_file in enumerate(self.files_list):
            data = np.load(self.path + "/" + self.files_list[batch_index])
            filesize = data["train_input"].shape[0]
            self.instances += filesize
        self.used_combinations = {}

    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        return math.floor(self.instances / self.batch_size)
    
    def __getitem__(self, index):
        for batch_index in range(self.batch_size):
            random_file_index = np.random.randint(0,len(self.files_list))
            filename = self.path + "/" + self.files_list[random_file_index]
            try:
                already_taken_indices = self.used_combinations[filename]
            except KeyError:
                self.used_combinations[filename] = []
                already_taken_indices = []            
            data = np.load(filename)
            filesize = data["train_input"].shape[0]
            #Select possible indices
            random_chunk_index = np.random.randint(0,filesize)
            #Register drawn index
            iterator=0
            while (random_chunk_index in already_taken_indices and iterator <= 10):
                random_chunk_index = np.random.randint(0,filesize)
                iterator += 1
            self.used_combinations[filename].append(random_chunk_index)
            new_x_train = data["train_input"][random_chunk_index]
            new_x_train2 = data["train_input2"][random_chunk_index]
            y_train = data["train_output"][random_chunk_index]
            self.X_train[batch_index,:,0] = new_x_train 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            self.X_train2[batch_index,:,:,0] = new_x_train2 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            self.labels[batch_index,:,:] = y_train 
            #Overwrites the zero initialized arrays in the constructor at specific positions
                
        X_total = {
            "Input_1": self.X_train,
            "Input_2": self.X_train2
            }
        return X_total,  self.labels
    
class Load_data_RNA_Validation(keras.utils.Sequence):

    """generate data in sequence mode for training the neural network"""
    def __init__(self, batch_size, path, files_list, labels):
        self.batch_size = batch_size
        self.path = path
        self.files_list = files_list
        self.labels_dim = labels  # Store labels dimension
        
        # Precompute all available samples
        self.data_indices = []  
        for file_idx, batch_file in enumerate(self.files_list):
            data = np.load(self.path + "/" + batch_file)
            filesize = data["train_input"].shape[0]
            self.data_indices.extend([(file_idx, i) for i in range(filesize)])

        self.instances = len(self.data_indices)  # Total instances in dataset

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return math.ceil(self.instances / self.batch_size)  # Ensure all data is used

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.data_indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_size_actual = len(batch_indices)  # Handle last batch case
        X_train = np.zeros([batch_size_actual, 400, 1])  
        X_train2 = np.zeros([batch_size_actual, 40, 4, 1])  
        labels = np.zeros([batch_size_actual, 40, self.labels_dim])  

        for batch_idx, (file_idx, sample_idx) in enumerate(batch_indices):
            filename = self.path + "/" + self.files_list[file_idx]
            data = np.load(filename)
            X_train[batch_idx, :, 0] = data["train_input"][sample_idx]
            X_train2[batch_idx, :, :, 0] = data["train_input2"][sample_idx]
            labels[batch_idx, :, :] = data["train_output"][sample_idx]

        X_total = {
            "Input_1": X_train,
            "Input_2": X_train2
        }
        return X_total, labels