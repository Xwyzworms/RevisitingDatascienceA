# %%
############################################################################################ REDef ALL Implementation ##########################################################################################################
### ###################################################
#       Coded by : Rose (Pratama Azmi A)
#       Date : 19/02/2023
##################################################
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Training information
VALIDATION_SIZE : float = 0.2
MIN_ACC : float= 0.86
# Embedding and Padding information
MAX_WORDS : int = 1000
OOV_TOKEN : str = "<OOV>"
MAX_LEN : int = 25
EMBEDDING_DIMS : int = 4

PADDING_TYPE : str = "post"
TRUNC_TYPE  : str = "post"


class Callback(keras.callbacks.Callback):
    
    
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch : %s acc %s val_Loss %s val_acc %s" %(epoch,logs.get("categorical_accuracy"),logs.get("val_loss"), logs.get("val_categorical_accuracy")))
        if(logs.get("val_categorical_accuracy") > MIN_ACC and logs.get("categorical_accuracy") >MIN_ACC) :
            print("Stopping Training ! Validation & train reach 0.86")
            self.model.stop_training = True

df : pd.DataFrame = pd.read_csv("EcoPreprocessed.csv")
## Convert to one hot encoding 

category_one_hot : pd.DataFrame = pd.get_dummies(df["division"])
df_oneHot = pd.concat([df, category_one_hot], axis=1)
df_oneHot.drop("division",axis=1,inplace=True)
X : np.array = df_oneHot["review"].values
Y : np.array = df_oneHot[["negative","neutral","positive"]].values


## Split the dataset and then go to Tokenizing & Padding
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=VALIDATION_SIZE)

tokenizer : Tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)

tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_test)

sequence_train : List[List[int]] = tokenizer.texts_to_sequences(x_train)
sequence_test : List[List[int]] = tokenizer.texts_to_sequences(x_test)

padded_sequence_train : List[List[int]] = pad_sequences(sequence_train, maxlen=MAX_LEN,
                                           padding=PADDING_TYPE, truncating=TRUNC_TYPE)

padded_sequence_test : List[List[int]] = pad_sequences(sequence_test, maxlen=MAX_LEN,
                                           padding=PADDING_TYPE, truncating=TRUNC_TYPE)
## PAdding it so its at same resolution
model : tf.keras.Sequential = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIMS),
    tf.keras.layers.LSTM(64, dropout=0.3),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

callbacksA = Callback()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=[
                    tf.keras.metrics.CategoricalAccuracy()
              ])
#### Modelling and Also Compiling
print("training begin")
history = model.fit(padded_sequence_train, y_train, validation_data=(padded_sequence_test, y_test), batch_size=128, verbose=0,epochs=100,callbacks=[callbacksA])

# %%
print(history.history.keys())
def plot_loss(history : pd.DataFrame):
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss","val_loss"])
    plt.show()

def plot_accuracy(history:pd.DataFrame):
    plt.plot(history["categorical_accuracy"])
    plt.plot(history["val_categorical_accuracy"])
    plt.title("accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("ACC")
    plt.show()


plot_loss(history.history)

# %%
plot_accuracy(history.history)


