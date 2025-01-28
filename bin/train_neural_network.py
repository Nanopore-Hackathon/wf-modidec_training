import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pod5
from remora import io, refine_signal_map
import tensorflow as tf
import plotly.graph_objects as go
import plotly
import json
import random

from keras.callbacks import LearningRateScheduler
from Load_data_for_training_V2 import Load_data_RNA # Data loader for efficient memory handling
from ModiDec_NN import ModiDeC_model  # Essential function to be imported
import os

opt_parser = argparse.ArgumentParser()

# Input options
opt_parser.add_argument(
    "-t",
    "--train_path",
    dest="train_path",
    help="Folder containing data for training the model",
    metavar="FILE",
)

opt_parser.add_argument(
    "-v",
    "--valid_path",
    dest="valid_path",
    help="Folder containing validation data for the model training",
    metavar="FILE",
)

opt_parser.add_argument(
    "-m",
    "--model_path",
    dest="model_path",
    help="TODO",
    metavar="FILE",
)

# Training options
opt_parser.add_argument(
    "-c",
    "--chunk_size",
    dest="chunk_size",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-b",
    "--batch_size",
    dest="batch_size",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-s",
    "--single_data_size",
    dest="single_data_size",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-l",
    "--max_seq_length",
    dest="max_seq_length",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-k",
    "--kmer_model",
    dest="kmer_model",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-y",
    "--labels",
    dest="labels",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    help="TODO",
    metavar="FILE",
)

opt_parser.add_argument(
    "-n",
    "--model_name",
    dest="model_name",
    help="TODO",
    metavar="FILE",
)

# Parse args
options = opt_parser.parse_args()

# Input options
train_path = str(options.train_path)
valid_path = str(options.valid_path)
model_path = str(options.model_path)

# Training options
batch_size = int(options.batch_size) #Batch size is the number of reads that should be used for training in parallel
kmer_model = int(options.kmer_model) #The k-mer size of the RNA kit
epochs = int(options.epochs)
model_name = str(options.model_name)


data_list = os.listdir(train_path)
probe_data = np.load(train_path + "/" + data_list[0])
probe_x1_data = probe_data["train_input"]
probe_y_data = probe_data["train_output"]

##All these parameters become defined during data curation
chunk_size = int(probe_x1_data.shape[1]) #Size of the receptive field of the network
single_data_size = int(probe_x1_data.shape[0])  #Single data size is a measure of how many reads are stored in a single npz file.
print("Singe data size: ",single_data_size)
labels = int(probe_y_data.shape[2]) # Number of labels
max_seq_length = int(probe_y_data.shape[1]) #Maximal length of sequences


# Inner function
# note: seq_len is chunk_size,
def NN_train(
    train_path: str,
    valid_path: str,
    model_path: str,
    seq_len: int,
    batch_size: int,
    single_data_size: int,
    max_seq_len: int,
    k_mer: int,
    labels: int,
    N_epoch: int,
    model_name: str,
):

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth: {tf.config.experimental.get_memory_growth(gpus[0])}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected")

    data_list = os.listdir(train_path)  # List of train data
    print(f"# Training files: {len(data_list)}")
    eval_list = os.listdir(valid_path)  # List of valid data
    print(f"# Validation files: {len(eval_list)}")
    #labels += 1

    N_batches = int(len(data_list) / (batch_size / single_data_size)) # The qoutient of batch_size/single_data_size defines how many files need to be read to collect enough reads to reach batch_size
    N_batches_2 = int(len(eval_list) / (batch_size / single_data_size))

    
    print(f"N Batches: {N_batches};N Batches 2: {N_batches_2}")

    # loads functions used for training
    # Data loading is performed by create a Load_data_RNA class. (Check file Load_data_for_training_V2.py)

    # Training dataset
    training_generator = Load_data_RNA(
        batch_size,
        N_batches,
        train_path,
        data_list,
        seq_len=seq_len,
        labels=labels,
        batch_loading=single_data_size,
        max_seq_len=max_seq_len
    )

    # Test dataset
    validation_generator = Load_data_RNA(
        batch_size,
        N_batches_2,
        valid_path,
        eval_list,
        seq_len=seq_len,
        labels=labels,
        batch_loading=single_data_size,
        max_seq_len=max_seq_len
    )

    # define the model
    # Creates the model by calling () function. (Import the file Inception_resnet_2inp_V2.py)
    model = ModiDeC_model(
        Inp_1=seq_len, Inp_2=max_seq_len, labels=labels, kmer_model=k_mer
    )

    # compile the model for the training
    # Optimizer of the neural network
    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=opt_adam, loss=tf.losses.binary_crossentropy, metrics=["accuracy"]
    )

    # Define the learning rate schedule
    def lr_schedule(epoch):
        min_lr = 0.00005  # Minimum learning rate
        initial_lr = 0.0001  # Example initial learning rate

        if epoch % 2 == 0 and epoch > 0:
            new_lr = initial_lr * (0.5 ** (epoch // 2))  # Exponential decay every 2 epochs
        else:
            new_lr = initial_lr

        # Ensure the learning rate doesn't go below the minimum
        return max(new_lr, min_lr)

    # Set up the LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(lr_schedule)    


    # starts the model training
    fit_results = model.fit(
        training_generator,
        validation_data=validation_generator,
        shuffle=True,
        epochs=N_epoch,
        workers=6,
        max_queue_size=256,
        callbacks=[lr_scheduler],
    )

    model.save(f"./{model_name}")
    print("training complete")
    return fit_results.history, validation_generator, model


# Outer function
def train_nn(
    train_path: str,
    valid_path: str,
    model_path: str,
    chunk_size: int,
    batch_size: int,
    single_data_size: int,
    max_seq_length: int,
    kmer_model: str,
    labels: int,
    epochs: int,
    model_name: str,
):
    # Call training and then plot some other stuff
    fit_results,validation_results, model = NN_train(
        train_path=train_path,
        valid_path=valid_path,
        model_path=model_path,
        seq_len=chunk_size,
        batch_size=batch_size,
        single_data_size=single_data_size,
        max_seq_len=max_seq_length,
        k_mer=kmer_model,
        labels=labels,
        N_epoch=epochs,
        model_name=model_name
    )
    # saves the model
    
    print(fit_results)

    layout = go.Layout(height=800)
    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(
            y=fit_results["accuracy"],
            mode="lines+markers",
            line=dict(color="rgba(72,99,156,1)"),
            showlegend=True,
            name="Training",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=fit_results["val_accuracy"],
            mode="lines+markers",
            line=dict(color="rgba(214,17,55,0.8)"),
            showlegend=True,
            name="Validation",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Iteration", gridcolor="white"),
        yaxis=dict(
            title="Accuracy", gridcolor="white", zeroline=True, zerolinecolor="black"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    plotly.io.write_html(fig, "./report_accuracy.html")

    # Plot loss
    layout = go.Layout(height=800)
    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(
            y=fit_results["loss"],
            mode="lines+markers",
            line=dict(color="rgba(72,99,156,1)"),
            showlegend=True,
            name="Training",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=fit_results["val_loss"],
            mode="lines+markers",
            line=dict(color="rgba(214,17,55,1)"),
            showlegend=True,
            name="Validation",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Iteration", gridcolor="white"),
        yaxis=dict(
            title="Loss", gridcolor="white", zeroline=True, zerolinecolor="black", range=[0,1]
        ),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    plotly.io.write_html(fig, "./report_loss.html")

    return fit_results


# train_nn()
train_nn(
    train_path=train_path,
    valid_path=valid_path,
    model_path=model_path,
    chunk_size=chunk_size,
    batch_size=batch_size,
    single_data_size=single_data_size,
    max_seq_length=max_seq_length,
    kmer_model=kmer_model,
    labels=labels,
    epochs=epochs,
    model_name=model_name,
)
