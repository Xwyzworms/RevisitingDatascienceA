# %%

#%%

################################################################################
# Coded by  : Rose ( Pratama Azmi A)
# Date : 23/02/2023
# Text Editor : Vscode + Vim
################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, LSTM, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from typing import List,Tuple
plt.rcParams["figure.figsize"] = (18,8)
plt.style.use("ggplot")


#%%

class Callback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch : %s ( Mae Train Loss -> %s &  Validation Loss -> %s )" %(epoch, logs["loss"],
                                                                                logs["val_loss"]))
        if(logs["val_loss"] < 0.10 and logs["loss"] < 0.10):
            self.model.stop_training= True
            print("Hit the validation MAE Loss <= 0.10")

    def on_predict_end(self, logs=None):
        print("Mae Loss for Validation Set is %s " %(logs["val_loss"]))
        
    


#%%
##### Utilities for visualizing
def printGeneralInformation(df : pd.DataFrame) -> None:
    print(df.info(),"\n\n") # Date is object dtype --> need to convert to datetime
    print(df.describe(),"\n\n")
    print("Null Value\n %s" %(df.isna().sum())) # 2204 Null

def plotIndexTrend(df : pd.DataFrame,indx_name : str ,listOfIndex : List[str]) -> None:

    for indx in listOfIndex:

        ## There's going 2 plot, the difference with previous day and also the close plot
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,8))
        ax[0].plot(df[df[indx_name] == indx]["Open"])
        ax[0].set_title("%s Open Daily close price" %(indx))
        
        ax[1].plot(df[df[indx_name] == indx]["Close"])
        ax[1].set_title("%s Daily Close price" %(indx))

        ax[2].plot(df[df[indx_name] == indx]["Close"].diff()) # Diff each day!
        ax[2].set_title("%s Daily Close price difference each day" %(indx))

        plt.show()

def printHighest(df : pd.DataFrame, columnName : str, n : int = 5) -> None :
    srt : pd.Series = df.sort_values(columnName, ascending= False)[columnName][:n]
    plt.scatter(srt.values, srt.index)
    plt.title("Highest for %s " %(columnName) + srt.to_string())
    plt.show()

def printLowest(df : pd.DataFrame, columnName : str, n : int = 5) -> None:
    srt : pd.Series = df.sort_values(columnName, ascending= True)[columnName][:n]
    plt.scatter(srt.values, srt.index)
    plt.title("Lowest for %s " %(columnName) + srt.to_string())
    plt.show()

def plot2LinePlots(df : pd.DataFrame, colName1 : str,
                   colName2 : str, titleP1 : str, titleP2 : str ) -> None:
    f, ax = plt.subplots(nrows=1, ncols=2,  figsize=(18, 8))
    ax[0].plot(df[colName1])
    ax[0].set_title(titleP1)

    ax[1].plot(df[colName2])
    ax[1].set_title(titleP2)


def printOpenCloseDifference(df : pd.DataFrame, 
                                    colName : str,
                                    n : int=5,
                                    asc : bool = False) -> None:
    printLowest(df,colName,n)
    printHighest(df,colName,n)

def plotRolling(df : pd.DataFrame, colName : str ,listOfRolling : List[int],
                 listOfLabels : List[str], title : str):
    for indx in range(0, len(listOfLabels)):
        plt.plot(df[colName].rolling(listOfRolling[indx]).mean(), label=listOfLabels[indx])
        plt.title(title)
        plt.legend()


def prepareTestingData(df : pd.DataFrame, colName : str , date_str : str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_size : int = len(df[df.index > date_str]) 
    train_dataset : pd.DataFrame = df[[colName]][:-test_size]
    test_dataset : pd.DataFrame = df[[colName]][-test_size:]

    return train_dataset, test_dataset

def prepareScaler_fit(train_dataset : pd.DataFrame, test_dataset: pd.DataFrame) -> Tuple[StandardScaler, np.array , np.array ] :
    train_standardScaler : StandardScaler = StandardScaler()

    train_dataset_scaled  : np.array = train_standardScaler.fit_transform(train_dataset)
    test_dataset_scaled : np.array = train_standardScaler.transform(test_dataset)

    return train_standardScaler, train_dataset_scaled, test_dataset_scaled


def defineModel(name : str) -> tf.keras.Sequential :

    model : tf.keras.Sequential = tf.keras.Sequential(name=name)
    model.add(LSTM(128, return_sequences= True, input_shape=(n_length, 1))) # (30, 1) Sesuai Batches 
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dense(1))


    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.MeanAbsoluteError()
    )

    return model


def printPlotComparison(scaler : StandardScaler, yTrue : pd.DataFrame,
                         yPred:pd.DataFrame, title : str) -> None:
    scaled_dat_yTrue : np.array = scaler.inverse_transform(yTrue)
    scaled_dat_yPred : np.array = scaler.inverse_transform(yPred)

    plt.plot(scaled_dat_yTrue , label="Real")
    plt.plot(scaled_dat_yPred, label="Prediction")
    plt.title(title)
    plt.legend()


#%%
df : pd.DataFrame = pd.read_csv("indexProcessed.csv")
printGeneralInformation(df)
#%%
print(df["Index"].unique()) # 
#%%
df["Date"] = pd.to_datetime(df["Date"])
# Making the date as the index 
df_date : pd.DataFrame = df.sort_values(["Index", "Date"]).set_index("Date")
print(df_date.head(2))
# %%
plotIndexTrend(df_date,"Index", df_date["Index"].unique())
#%%

# Exploratory Data Analysis for GDAXI
gdaxi_df : pd.DataFrame = df_date[df_date["Index"] == "GDAXI"]
gdaxi_df["Open_Close"] = gdaxi_df["Close"].values - gdaxi_df["Open"].values
print(gdaxi_df.isna().sum())
plot2LinePlots(gdaxi_df, "Open", "Open_Close", "Gdaxi Open Daily Price", "Gdaxi Open-Close daily Price")
# %%
n : int = 100
printHighest(gdaxi_df, "Open",n)
printHighest(gdaxi_df, "Close",n)

printLowest(gdaxi_df, "Open",n)
printLowest(gdaxi_df, "Close",n)

printOpenCloseDifference(gdaxi_df, "Open_Close",n)

# %%
rollingInfo : List[int] = [30,60,90]
labelsInfo : List[str] = ["30 days", "60 days", "90 days"]
plotRolling(gdaxi_df,"Close", rollingInfo, labelsInfo, "GDXAII CLOSE ")
#%%

gdaxi_train, gdaxi_test = prepareTestingData(gdaxi_df, "Close", "2014-01-01")

plt.plot(gdaxi_test, label="After 2019", color="black")
plt.plot(gdaxi_train, label="Before 2019", color="red")
plt.legend()
plt.grid()
plt.title("Gdaxi Stock Change")
plt.show()
#%%
##3 Datapreparation
gdaxi_scaler, gdaxi_train_scaled, gdaxi_test_scaled = prepareScaler_fit(gdaxi_train, gdaxi_test)

print(gdaxi_train_scaled.shape)
print(gdaxi_test_scaled.shape)

## TimeStep = 30 day
n_length : int = 30
## Each batch has 30 day !
train_gdaxi_generator : TimeseriesGenerator = TimeseriesGenerator(gdaxi_train_scaled,gdaxi_train_scaled, length=n_length)
test_gdaxi_generator : TimeseriesGenerator = TimeseriesGenerator(gdaxi_test_scaled, gdaxi_test_scaled,length=n_length )

# Modelling
model_gdaxi : tf.keras.Model = defineModel("GDAXI")
EPOCHS : int = 100
model_gdaxi_history : dict = model_gdaxi.fit(
    train_gdaxi_generator,
    validation_data=(test_gdaxi_generator),
    batch_size=1, epochs=EPOCHS, verbose =0,callbacks=[Callback()])
 # BATCH_SIZE 1, because we already batched it on TIMERSERIESGENERATOR
#%%
prediction = model_gdaxi.predict(test_gdaxi_generator)
printPlotComparison(gdaxi_scaler, prediction, gdaxi_test_scaled, "GDAXI Daily Close Price : Real vs Pred")


# %%
def plotHistoryLoss(history : dict) -> None:
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss","val_loss"])
    plt.show()


plotHistoryLoss(model_gdaxi_history)

# %% [markdown]
# plotHistoryLoss(model_gdaxi_history)


