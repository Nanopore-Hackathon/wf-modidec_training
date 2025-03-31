
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

    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        return math.floor(self.instances / self.batch_size)
    
    def __getitem__(self, index):
        iterator = 0
        for batch_index in range(self.batch_size):
            if iterator % math.ceil(self.batch_size * 0.2) == 0:
                random_file_index = np.random.randint(0,len(self.files_list))
                filename = self.path + "/" + self.files_list[random_file_index]       
                data = np.load(filename)
                filesize = data["train_input"].shape[0]
                random_chunk_index = np.random.randint(0,filesize - 16)
                random_chunk_index = random_chunk_index - (random_chunk_index % 16)
            #Select possible indices
            modulo_index = iterator % 16
            new_x_train = data["train_input"][random_chunk_index + modulo_index]
            new_x_train2 = data["train_input2"][random_chunk_index + modulo_index]
            y_train = data["train_output"][random_chunk_index + modulo_index]
            self.X_train[batch_index,:,0] = new_x_train 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            self.X_train2[batch_index,:,:,0] = new_x_train2 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            self.labels[batch_index,:,:] = y_train 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            iterator += 1
                
        X_total = {
            "Input_1": self.X_train,
            "Input_2": self.X_train2
            }
        return X_total,  self.labels
    
    
class Load_data_RNA_Validation(keras.utils.Sequence):

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

    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        return math.floor(self.instances / (self.batch_size * 3))
    
    def __getitem__(self, index):
        iterator = 0
        for batch_index in range(self.batch_size):
            if iterator % math.ceil(self.batch_size * 0.2) == 0:
                random_file_index = np.random.randint(0,len(self.files_list))
                filename = self.path + "/" + self.files_list[random_file_index]       
                data = np.load(filename)
                filesize = data["train_input"].shape[0]
                random_chunk_index = np.random.randint(0,filesize - 16)
                random_chunk_index = random_chunk_index - (random_chunk_index % 16)
            #Select possible indices
            modulo_index = iterator % 16
            new_x_train = data["train_input"][random_chunk_index + modulo_index]
            new_x_train2 = data["train_input2"][random_chunk_index + modulo_index]
            y_train = data["train_output"][random_chunk_index + modulo_index]
            self.X_train[batch_index,:,0] = new_x_train 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            self.X_train2[batch_index,:,:,0] = new_x_train2 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            self.labels[batch_index,:,:] = y_train 
            #Overwrites the zero initialized arrays in the constructor at specific positions
            iterator += 1
                
        X_total = {
            "Input_1": self.X_train,
            "Input_2": self.X_train2
            }
        return X_total,  self.labels