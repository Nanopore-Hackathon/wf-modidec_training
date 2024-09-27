import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pod5
from remora import io, refine_signal_map
import tensorflow as tf
import plotly.graph_objects as go
import plotly

from keras.callbacks import LearningRateScheduler
from Load_data_for_training_V2 import (
    Load_data_RNA,
)  # Data loader for efficient memory handling
from PseudoDeC_NN import PseudoDec_NN_Model  # Essential function to be imported
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
chunk_size = int(options.chunk_size)
batch_size = int(options.batch_size)
single_data_size = int(options.single_data_size)
max_seq_length = int(options.max_seq_length)
kmer_model = int(options.kmer_model)
labels = int(options.labels)
epochs = int(options.epochs)
model_name = str(options.model_name)


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
    eval_list = os.listdir(valid_path)  # List of valid data

    labels += 1

    N_batches = int(len(data_list) / (batch_size / single_data_size))
    N_batches_2 = int(len(eval_list) / (batch_size / single_data_size))

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
        max_seq_len=max_seq_len,
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
        max_seq_len=max_seq_len,
    )

    # define the model
    # Creates the model by calling () function. (Import the file Inception_resnet_2inp_V2.py)
    model = PseudoDec_NN_Model(
        Inp_1=seq_len, Inp_2=max_seq_len, labels=labels, kmer_model=k_mer
    )

    # compile the model for the training
    # Optimizer of the neural network
    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=opt_adam, loss=tf.losses.binary_crossentropy, metrics=["accuracy"]
    )

    # Learning rate (lr) scheduler
    # Varies learning rate after each epoch
    def lr_schedule(epoch, optimizer):

        min_lr = 0.0000125  # Set the minimum learning rate

        # Update the learning rate if needed (similar to your original code)
        if epoch % 2 == 0 and epoch > 0:

            new_lr = (
                tf.keras.backend.get_value(model.optimizer.lr) * 0.5
            )  # You can adjust the decay factor as needed
            model.optimizer.lr.assign(new_lr)
            return max(new_lr, min_lr)

        else:
            return tf.keras.backend.get_value(model.optimizer.lr)

    lr_scheduler = LearningRateScheduler(
        lambda epoch: lr_schedule(epoch, optimizer=opt_adam)
    )

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

    # saves the model
    model.save(model_path + "/" + model_name)

    print("training complete")
    return fit_results


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
    fit_results = NN_train(
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
        model_name=model_name,
    )

    # print("Plotting...")
    # # Plot accuracy
    # layout = go.Layout(height=800)
    # fig = go.Figure(layout=layout)

    # fig.add_trace(
    #     go.Scatter(
    #         y=fit_results["acc"],
    #         mode="lines+markers",
    #         line=dict(color="rgba(72,99,156,1)"),
    #         showlegend=True,
    #         name="Training",
    #     )
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         y=fit_results["val_acc"],
    #         mode="lines+markers",
    #         line=dict(color="rgba(19,24,156,1)"),
    #         showlegend=True,
    #         name="Validation",
    #     )
    # )

    # fig.update_layout(
    #     xaxis=dict(title="Iteration", gridcolor="white"),
    #     yaxis=dict(
    #         title="Accuracy", gridcolor="white", zeroline=True, zerolinecolor="black"
    #     ),
    #     plot_bgcolor="rgba(0,0,0,0)",
    # )
    # plotly.io.write_html(fig, "./report_accuracy.html")

    # # Plot loss
    # layout = go.Layout(height=800)
    # fig = go.Figure(layout=layout)

    # fig.add_trace(
    #     go.Scatter(
    #         y=fit_results["loss"],
    #         mode="lines+markers",
    #         line=dict(color="rgba(72,99,156,1)"),
    #         showlegend=True,
    #         name="Training",
    #     )
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         y=fit_results["val_loss"],
    #         mode="lines+markers",
    #         line=dict(color="rgba(19,24,156,1)"),
    #         showlegend=True,
    #         name="Validation",
    #     )
    # )

    # fig.update_layout(
    #     xaxis=dict(title="Iteration", gridcolor="white"),
    #     yaxis=dict(
    #         title="Loss", gridcolor="white", zeroline=True, zerolinecolor="black"
    #     ),
    #     plot_bgcolor="rgba(0,0,0,0)",
    # )
    # plotly.io.write_html(fig, "./report_loss.html")

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
