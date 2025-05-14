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
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall

from keras.callbacks import LearningRateScheduler
from Load_data_for_training_V2 import Load_data_RNA,Load_data_RNA_Validation # Data loader for efficient memory handling
from ModiDec_NN import ModiDeC_model  # Essential function to be imported
import os

from sklearn.metrics import roc_curve, auc

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

opt_parser.add_argument(
    "-b",
    "--batch_size",
    dest="batch_size",
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
labels = int(probe_y_data.shape[2]) # Number of labels


# Inner function
def NN_train(
    train_path: str,
    valid_path: str,
    model_path: str,
    batch_size: int,
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

    # loads functions used for training
    # Data loading is performed by create a Load_data_RNA class. (Check file Load_data_for_training_V2.py)

    # Training dataset
    training_generator = Load_data_RNA(
        batch_size = batch_size,
        path = train_path,
        files_list = data_list,
        labels=labels
    )

    # Test dataset
    validation_generator = Load_data_RNA_Validation(
        batch_size = batch_size,
        path = valid_path,
        files_list = eval_list,
        labels = labels
    )

    # define the model
    # Creates the model by calling () function. (Import the file Inception_resnet_2inp_V2.py)
    model = ModiDeC_model(
        Inp_1=400, Inp_2=40, labels=labels, kmer_model=k_mer
    )

    # compile the model for the training
    # Optimizer of the neural network
    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    

    # def f1_score(y_true, y_pred):
    #     precision = Precision()(y_true, y_pred)
    #     recall = Recall()(y_true, y_pred)
    #     return 2 * (precision * recall) / (precision + recall + K.epsilon())

    # model.compile(
    #     optimizer=opt_adam, loss=tf.losses.binary_crossentropy, metrics=["accuracy",Precision(name="precision"),Recall(name="recall")]
    # )
    
    
    class_weights = tf.constant([0.6 if index_mod == 0 else 1.0 for index_mod in range(probe_data["train_output"].shape[2])])  # Replace with computed weights

    # def weighted_categorical_crossentropy(y_true, y_pred):
    #     ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    #     weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    #     return ce * weights  # Apply weight per sample
    
    # model.compile(
    #     optimizer=opt_adam, loss=weighted_categorical_crossentropy, metrics=["accuracy",Precision(name="precision"),Recall(name="recall")]
    # )
    
    
    def weighted_binary_crossentropy(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # Multiply loss by feature-wise weights
        weighted_bce = bce * tf.reshape(class_weights, (probe_data["train_output"].shape[2], probe_data["train_output"].shape[1]))
        return weighted_bce  # Keeps shape consistent
    
    model.compile(
        optimizer=opt_adam, loss=weighted_binary_crossentropy, metrics=["accuracy",Precision(name="precision"),Recall(name="recall")]
    )

    
    # Define the learning rate schedule
    def lr_scheduler(epoch):
        min_lr = 0.0000125  # Minimum learning rate
        initial_lr = 0.0001  # Example initial learning rate

        if (epoch + 1) % 2 == 0 and epoch > 0:
            new_lr = initial_lr * (0.5 ** (epoch // 2))  # Exponential decay every 2 epochs
        else:
            new_lr = initial_lr

        # Ensure the learning rate doesn't go below the minimum
        return max(new_lr, min_lr)

    # Set up the LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(lr_scheduler)    


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
    
    # Get the true labels and predictions from validation generator
    
    print("training complete")
    return fit_results.history, validation_generator, model


# Outer function
def train_nn(
    train_path: str,
    valid_path: str,
    model_path: str,
    batch_size: int,
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
        batch_size=batch_size,
        k_mer=kmer_model,
        labels=labels,
        N_epoch=epochs,
        model_name=model_name
    )
    # saves the model
    
    print(fit_results)
    def create_lineplot(fit_results,metric:str):
        layout = go.Layout(height=800)
        fig = go.Figure(layout=layout)
        x_axis = np.arange(1, len(fit_results["accuracy"]) + 1, 1)

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y= fit_results[f"{metric}"],
                mode="lines+markers",
                line=dict(color="rgba(72,99,156,1)"),
                showlegend=True,
                name="Training",
            )
        )

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y=fit_results[f"val_{metric}"],
                mode="lines+markers",
                line=dict(color="rgba(214,17,55,0.8)"),
                showlegend=True,
                name="Validation",
            )
        )

        fig.update_layout(
            xaxis=dict(title="Iteration", gridcolor="white"),
            yaxis=dict(
                title=f"{metric.capitalize()}", gridcolor="white", zeroline=True, zerolinecolor="black"
            ),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        plotly.io.write_html(fig, f"./report_{metric}.html")

    create_lineplot(fit_results, "accuracy")
    create_lineplot(fit_results,"precision")
    create_lineplot(fit_results,"recall")
    create_lineplot(fit_results,"loss")
    
    def create_roc_curve(validation_results, model):
        colors = [
        "rgba(255, 179, 186, 1)", "rgba(255, 223, 186, 1)", "rgba(255, 255, 186, 1)", "rgba(186, 255, 201, 1)",
        "rgba(186, 225, 255, 1)", "rgba(255, 186, 239, 1)", "rgba(217, 217, 217, 1)", "rgba(219, 112, 147, 1)",
        "rgba(218, 165, 32, 1)", "rgba(144, 238, 144, 1)", "rgba(173, 216, 230, 1)", "rgba(216, 191, 216, 1)",
        "rgba(255, 182, 193, 1)", "rgba(255, 218, 185, 1)", "rgba(175, 238, 238, 1)", "rgba(250, 250, 210, 1)",
        "rgba(245, 245, 220, 1)", "rgba(255, 228, 196, 1)", "rgba(230, 230, 250, 1)", "rgba(240, 255, 255, 1)",
        "rgba(240, 255, 240, 1)", "rgba(255, 250, 250, 1)", "rgba(255, 245, 238, 1)", "rgba(255, 239, 213, 1)",
        "rgba(255, 228, 225, 1)", "rgba(240, 248, 255, 1)", "rgba(224, 255, 255, 1)", "rgba(240, 230, 140, 1)",
        "rgba(245, 222, 179, 1)", "rgba(176, 224, 230, 1)", "rgba(255, 192, 203, 1)", "rgba(152, 251, 152, 1)",
        "rgba(221, 160, 221, 1)", "rgba(255, 160, 122, 1)", "rgba(135, 206, 250, 1)", "rgba(100, 149, 237, 1)",
        "rgba(102, 205, 170, 1)", "rgba(123, 104, 238, 1)", "rgba(147, 112, 219, 1)", "rgba(186, 85, 211, 1)",
        "rgba(199, 21, 133, 1)", "rgba(139, 0, 139, 1)", "rgba(0, 255, 127, 1)", "rgba(127, 255, 212, 1)",
        "rgba(255, 105, 180, 1)", "rgba(255, 20, 147, 1)", "rgba(219, 112, 147, 1)", "rgba(255, 228, 181, 1)",
        "rgba(210, 180, 140, 1)", "rgba(255, 239, 0, 1)", "rgba(255, 182, 193, 1)", "rgba(221, 160, 221, 1)",
        "rgba(176, 224, 230, 1)", "rgba(144, 238, 144, 1)", "rgba(255, 160, 122, 1)", "rgba(240, 128, 128, 1)",
        "rgba(255, 127, 80, 1)", "rgba(250, 128, 114, 1)", "rgba(233, 150, 122, 1)", "rgba(216, 112, 147, 1)",
        "rgba(255, 69, 0, 1)", "rgba(255, 140, 0, 1)", "rgba(255, 215, 0, 1)", "rgba(173, 255, 47, 1)",
        "rgba(124, 252, 0, 1)", "rgba(127, 255, 0, 1)", "rgba(0, 250, 154, 1)", "rgba(32, 178, 170, 1)",
        "rgba(72, 209, 204, 1)", "rgba(0, 206, 209, 1)", "rgba(95, 158, 160, 1)", "rgba(70, 130, 180, 1)",
        "rgba(30, 144, 255, 1)", "rgba(0, 191, 255, 1)", "rgba(65, 105, 225, 1)", "rgba(138, 43, 226, 1)",
        "rgba(139, 69, 19, 1)", "rgba(160, 82, 45, 1)", "rgba(210, 105, 30, 1)", "rgba(205, 133, 63, 1)",
        "rgba(244, 164, 96, 1)", "rgba(255, 228, 181, 1)", "rgba(255, 222, 173, 1)", "rgba(255, 218, 185, 1)",
        "rgba(255, 250, 205, 1)", "rgba(240, 230, 140, 1)", "rgba(255, 239, 213, 1)", "rgba(245, 245, 220, 1)"
        ]
        y_true = []
        y_scores = []
        for batch, labels in validation_results:
            y_true.extend(labels)  # Store true labels
            y_scores.extend(model.predict(batch))  # Store predicted probabilities
        
        layout = go.Layout(height=800)
        fig = go.Figure(layout=layout)
        
        for mod_index in range(y_scores[0].shape[1]):
            temp_y_true = np.zeros(40*len(y_true))
            temp_y_scores = np.zeros(40*len(y_scores))
            for output_index, (y_scores_i,y_true_i) in enumerate(zip(y_scores,y_true)): 
                temp_y_true[output_index*40:(output_index + 1)*40] = y_true_i[:,mod_index]
                temp_y_scores[output_index*40:(output_index + 1)*40] = y_scores_i[:,mod_index]
            temp_fpr, temp_tpr, _ = roc_curve(temp_y_true,temp_y_scores)
            temp_roc_auc = auc(temp_fpr,temp_tpr)
            mod_type = str(mod_index)
            if mod_index == 0:
                mod_type = "Unmodified"
            fig.add_trace(
            go.Scatter(
                x = temp_fpr,
                y= temp_tpr,
                mode="lines",
                line=dict(color=colors[mod_index]),
                showlegend=True,
                name=f"Mod type: {mod_type}, AUC: {temp_roc_auc}",
                )
            )
        fig.update_layout(
        xaxis=dict(title="FPR", gridcolor="white"),
        yaxis=dict(
            title="TPR", gridcolor="white", zeroline=True, zerolinecolor="black"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        )
        plotly.io.write_html(fig, "./report_roc.html")    
    create_roc_curve(validation_results, model)
    return fit_results


train_nn(
    train_path=train_path,
    valid_path=valid_path,
    model_path=model_path,
    batch_size=batch_size,
    kmer_model=kmer_model,
    labels=labels,
    epochs=epochs,
    model_name=model_name,
)
