#%%

################################################################################
# Coded by  : Rose ( Pratama Azmi A)
# Date : 21/02/2023
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
from keras.layers import Dense, LSTM
from keras.models import Sequential
from typing import List
plt.rcParams["figure.figsize"] = (18,8)
plt.style.use("ggplot")

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

def printHighest(df : pd.DataFrame, columnName : str, n : int = 5):
    srt : pd.Series = df.sort_values(columnName, ascending= False)[columnName][:n]
    plt.scatter(srt.values, srt.index)
    plt.title("Highest for %s " %(columnName) + srt.to_string())
    plt.show()

def printLowest(df : pd.DataFrame, columnName : str, n : int = 5):
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


#%%
df : pd.DataFrame = pd.read_csv("indexData.csv")
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

# Predict
gdaxi_df : pd.DataFrame = df_date[df_date["Index"] == "GDAXI"]
gdaxi_df.sort_values("Close", ascending=False)["Close"][:3]
gdaxi_df["Open_Close"] = gdaxi_df["Close"].values - gdaxi_df["Open"].values

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
