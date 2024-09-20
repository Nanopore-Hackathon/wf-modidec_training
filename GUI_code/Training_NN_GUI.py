import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QLabel, QHBoxLayout, QCheckBox
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from Load_data_for_training_V2 import  Load_data_RNA #Data loader for efficient memory handling
from PseudoDeC_NN import PseudoDec_NN_Model #Essential function to be imported 
import os

class MainWindow(QMainWindow):
    def __init__(self):
        """
        Init function to create the GUI surface and definition of important inputs for the pipeline
        Folders that are defined here are:
        
        training_data_folder -> folder1
        validation_data_folder -> folder2
        output_folder -> folder3
        """
        super().__init__()

        # list of variables
        self.paths = {"folder1": None, "folder2": None , "folder3": None}

        # Set up the main window
        self.setWindowTitle("Training Nueral network - modification classifier")
        self.setGeometry(100, 100, 320, 100)

        # Create a QWidget and set it as the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a vertical layout
        layout = QVBoxLayout()

        #Variable self.button2 defines the directory for training data
        #Gives button a Widget identity and a title for the gui
        self.button1 = QPushButton('training data folder')
        #Once a button is pushed by the user this line open a file dialogue to select the training data folder
        # The selected flder is directly imported to self.paths as self.paths["folder1"]
        self.button1.clicked.connect(lambda: self.open_directory_dialog('folder1'))
        #Places widget on the GUI grid
        layout.addWidget(self.button1)

        #Variable self.button2 defines the directory for validation data
        #Gives button a Widget identity and a title for the gui
        self.button2 = QPushButton('Validation data folder')
        #Once a button is pushed by the user this line open a file dialogue to select the validation data folder
        # The selected flder is directly imported to self.paths as self.paths["folder2"]
        self.button2.clicked.connect(lambda: self.open_directory_dialog('folder2'))
        #Places widget on the GUI grid
        layout.addWidget(self.button2)
        
        
        #Variable self.button3 defines the output directory of the network training
        #Gives button a Widget identity and a title for the gui
        self.button3 = QPushButton('save model folder') 
        #Once a button is pushed by the user this line open a file dialogue to select the output folder
        # The selected flder is directly imported to self.paths as self.paths["folder3"]
        self.button3.clicked.connect(lambda: self.open_directory_dialog('folder3')) 
        #Places widget on the GUI grid                                                                            
        layout.addWidget(self.button3) 

        # set the first set of variables
        textbox1 = QLabel("General variables for training data:")
        layout.addWidget(textbox1)
        self.setup_variables(layout)

        #Variable self.button4 defines the start button for the model training
        #Gives button a Widget identity and a title for the gui
        self.button4 = QPushButton('Start training')
        #Links the action of pushing the button to the initiation of the function of the model training 
        self.button4.clicked.connect(lambda: self.start_training())
        layout.addWidget(self.button4)

        # Set the layout on the central widget
        self.central_widget.setLayout(layout)


    """ list of function used in the main"""

    def open_directory_dialog(self, folder_name):
        """
            Function opens a dialog window to select folders and overwrite
            the self.paths dictionairy with costum directory paths:
            
            training_data_folder -> folder1 -> self.paths["folder1"]
            validation_data_folder -> folder2 -> self.paths["folder2"]
            output_folder -> folder3 -> self.paths["folder3"]
        """
        # Open a dialog to choose a directory
        directory = QFileDialog.getExistingDirectory(self, f"Select {folder_name}")
        if directory:
            self.paths[folder_name] = directory
            print(f"Selected path for {folder_name}: {directory}")


    def setup_variables(self, layout):
        """
        This function creates a GUI surface to determine customizable 
        processing variables, which need to be defined before the model training:
        
        chunk_size:int -> How many timepoints taken for a sliding window on the raw signal at once. Default value: 400 (Should not be changed)   
        batch_size:int -> How many datasets should be analyzed in parallel. Default:256 (Depending on GPU capacity)
        single_data_size:int -> How many files should be stored in one numpy zip ouput file ? Default: 16 
        max_seq_length:int -> The maximum length of a sequence for the given chunksize. Default: 40 (for chunksize 400 is stable, rule of thumb: chunksize / 10 )
        kmer_model:int -> How many bases are the kmers of your data containing ? (RNA004: 9, RNA002: 5)  
        labels:int -> How many modifications do you want to analyze ? Default: 1 modification type 
        epochs:int -> How many epochs/rounds do you want to train the network ?
        name_NN:str -> Output name of your trained network model
        
        Function creates a QLineEdit Widget for every variable and takes the input of the entries to these lines
        as variables for the training of the neural network. First they will be read as strings, but they become casted in
        the start_training function.
        """
        # Creating layout for each variable in Variables tuple
        labels = ["chunck_size (int)", "batch_size (int)", 
                  "single_data_size (int)", "max seq. length (int)", "k-mer model (int)",
                  "labels (int)", "epoches (suggeste 4) (int)", "name NN (str)" ]
        
        self.vars_entries = []
        #Loop to create Widgets for each variable and place them in the GUI
        #Inputs will be provided by the user when running the GUI and writing on the QLineEdit widgets
        for i, label in enumerate(labels):
            row_layout = QHBoxLayout()
            label_widget = QLabel(label + ":")
            input_widget = QLineEdit()
            row_layout.addWidget(label_widget)
            row_layout.addWidget(input_widget)
            layout.addLayout(row_layout)
            self.vars_entries.append(input_widget)

    def start_training(self):
        """
        Main function to run the network training
        """

        #Loads all variables which are provided within the running GUI by the user and casts them to obtain the right datatypes.
        path_data = self.paths["folder1"]
        data_list = os.listdir(path_data)


        path_eval = self.paths["folder2"]
        eval_list = os.listdir(path_data)

        seq_len = int( self.vars_entries[0].text()) # chunck_size
        batch_size = int( self.vars_entries[1].text())
        single_data_size = int( self.vars_entries[2].text())
        max_seq_len = int( self.vars_entries[3].text())
        k_mer = int( self.vars_entries[4].text())
        labels = int( self.vars_entries[5].text()) + 1
        N_epoch = int( self.vars_entries[6].text())

        N_batches = int(len(data_list)/(batch_size/single_data_size))
        N_batches_2 = int(len(eval_list)/(batch_size/single_data_size))

        #loads functions used for training
        #Data loading is performed by create a Load_data_RNA class. (Check file Load_data_for_training_V2.py)
        
        #Training dataset
        training_generator =  Load_data_RNA(batch_size, N_batches,
                                            path_data, 
                                            data_list, seq_len = seq_len, 
                                            labels= labels , 
                                            batch_loading = single_data_size,
                                            max_seq_len= max_seq_len)

        #Test dataset
        validation_generator =  Load_data_RNA(batch_size, N_batches_2,
                                            path_eval, 
                                            eval_list, seq_len = seq_len, 
                                            labels= labels , 
                                            batch_loading = single_data_size,
                                            max_seq_len= max_seq_len)


        #define the model
        # Creates the model by calling () function. (Import the file Inception_resnet_2inp_V2.py)
        model = PseudoDec_NN_Model(Inp_1 = seq_len, Inp_2 = max_seq_len, labels = labels, kmer_model=k_mer)


        #compile the model for the training
        #Optimizer of the neural network
        opt_adam =tf.keras.optimizers.Adam(learning_rate= 0.0001)

        model.compile(optimizer=opt_adam, 
                loss= tf.losses.binary_crossentropy, 
                metrics=["accuracy"])

        #Learning rate (lr) scheduler
        #Varies learning rate after each epoch
        def lr_schedule(epoch, optimizer):

            min_lr = 0.0000125  # Set the minimum learning rate

            # Update the learning rate if needed (similar to your original code)       
            if epoch % 2 == 0 and epoch > 0:

                new_lr = tf.keras.backend.get_value(model.optimizer.lr) * 0.5  # You can adjust the decay factor as needed
                model.optimizer.lr.assign(new_lr)
                return max(new_lr, min_lr)
            
            else:
                return tf.keras.backend.get_value(model.optimizer.lr)
    
        lr_scheduler = LearningRateScheduler(lambda epoch: lr_schedule(epoch, optimizer=opt_adam))

        #starts the model training
        model.fit(training_generator, 
                    shuffle = True, 
                    epochs=N_epoch, 
                    workers= 6, 
                    max_queue_size=256,
                    callbacks= [lr_scheduler]) # callbacks= [lr_scheduler]

        #saves the model
        model.save( self.paths["folder3"] + "/" + self.vars_entries[7].text())

        print("training complete")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
