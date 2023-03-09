### ###################################################
#       Coded by : Rose (Pratama Azmi A)
#       Date : Unknown 
#       Text editor : Vscode + VIM
#       Credits to Mr.Didit Aditya  
#       Static typing version 
##################################################

#%%
### Prepare plotting information
## Static Typing 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict
from matplotlib import rcParams # Runtime configuration
rcParams["figure.figsize"] = (14,7)
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

#%%
class SimpleLinearRegression():
    """
        Class to implements simple linear regression model
        using calculus approach not COVARIANCE !
    """
    def __init__(self):
        """ 
            b0 : Intercept, the baseline for current study case
            b1 : basically the gradient/weight/bobot for coresponding feature
        """
        self.b0 : float = None
        self.b1 : float = None 


    def fit(self, x : np.array , y : np.array )-> None:
        """
           Calculate the SLOPE and Intercept Coefficients : y = b0 + b1x

        Args:
            x (np.array): single feature !
            y (np.array): True value 
            return None
        """

        numerator : float = np.sum(np.dot(x,(y-np.mean(y) )))  
        denominator : float = np.sum(np.dot(x,(x - np.mean(x))))
        
        self.b1 = numerator / denominator
        self.b0 = np.mean(y) - self.b1 * np.mean(x)
            
    def predict(self, x : np.array) -> float:
        """ 
            Predict using  line equation 
        """
        # Check if the model already fitted or not
        if not self.b0 or not self.b1:
            raise Exception("Please fit the data first !, call SimpleLinearRegression.fit(x,y)")
        return self.b0 + self.b1*x




#%%
##############################################################################################################################################
## Plotting dummy data 


x : np.array = np.arange(start =1, stop = 301) # contains np arr from 1 to 400
y : np.array = np.random.normal(loc=x, scale=20) # contains random normal distribution values centered around x with std 20 
print(np.mean(y))
# Plot
plt.scatter(x,y, s = 200, c='#333333', alpha=0.65)
plt.title("Source data", size = 20)
plt.xlabel("X", size= 10)
plt.ylabel("Y",  size=10)
plt.show()
#%% Fitting the model 
# it should e validation set, but nevermind
# 20%
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Modelling process OR optimization Process, anjay
model : SimpleLinearRegression = SimpleLinearRegression()
model.fit(x_train,y_train)
preds : np.array = model.predict(x_test)

print(model.b0, model.b1)

# %%
## Lets Validate the model
# 1> Qualitatively --> Visualization
# 2> Quantitative --> RMSE, MAE, MSE

########################## 1. Qualitative Method #################################
model_entire : SimpleLinearRegression = SimpleLinearRegression()
model_entire.fit(x,y)
preds_entire : np.array = model_entire.predict(x)
#%%
plt.scatter(x,y, s=200, c="#087E8B", alpha=0.65, label="Source Data")
plt.plot( x, preds_entire,
          color="#333333", lw=3, label=f"best fit B0: {model_entire.b0:.2f} B1: {model_entire.b1:.2f}")
plt.legend()
plt.title("Best fit Line : Least Square Method" , size=20)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

########################## 1. Quantitative Method #################################
rmse : float = np.sqrt(mean_squared_error(y_test,preds))
print(rmse)

### So on average the different between the predicted with yTrue is 20.093 kalau misalnya di
### analogikan pada studi kasus taksi dengan satuan dollar, perbedaannya mungkin bisa 20dollar ( RIP )


#%%
# Comparasion with the sklearn model
from sklearn.linear_model import LinearRegression
sk_model : LinearRegression = LinearRegression()
sk_model.fit(np.array(x_train).reshape(-1,1), y_train)
sk_preds : np.array = sk_model.predict(np.array(x_test).reshape(-1,1))
print(sk_model.intercept_, sk_model.coef_)
print(np.sqrt(mean_squared_error(y_test, sk_preds)))

### The result should be the same !
### I love mathematics ! Let Continue again
##################################################################################################################################################################
# %%
#### Simple Analysis for Housing dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (14,8)
from sklearn.model_selection import train_test_split 
df : pd.DataFrame = pd.read_csv("housing.csv")
df.head()

print(df.isna().sum())

def printGeneralInformation(df : pd.DataFrame) :
    print(df.describe())
    print(df.var())

def drawScatterPlot(df, x : str ,y : str, title : str) -> None:
    sns.scatterplot(data=df,y=df[x],x=df[y])
    plt.title(title)
    plt.show()

def drawBarplot(df, x : str, y : str, title: str):
    sns.barplot(data=df,y=df[x],x=df[y])
    plt.title(title)
    plt.show()
printGeneralInformation(df)


#%%
sns.heatmap(df.corr(), cmap="BuGn", annot=True, fmt=".2f")

#%% 
## Draw Scatter plot for Area 
drawScatterPlot(df,"price","area","Area's Effect to Prices")

#%%
x :np.array = df["area"].values
y : np.array = df["price"].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
simpleLin : SimpleLinearRegression() = SimpleLinearRegression()
simpleLin.fit(x_train,y_train)
preds = simpleLin.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,preds)))

#%%

y_train_preds = simpleLin.predict(x_train)
plt.scatter(x_train,y_train, label="Train set ")
plt.plot(x_train, y_train_preds, color="#333333", label=f"Intercept : {simpleLin.b0:.2f} Area Gradient : {simpleLin.b1:.2f}" )
plt.legend()
plt.xlabel("Area")
plt.title("Train set result")


#%%
plt.scatter(x_test,y_test, label="Test set ")
plt.plot(x_test, preds, color="#333333", label=f"Intercept : {simpleLin.b0:.2f} Area Gradient : {simpleLin.b1:.2f}" )
plt.legend()
plt.xlabel("Area")
plt.title("Test set result")

# %%
print(simpleLin.b0)
print(simpleLin.b1)
#%%
df["area"]
#%%
df[df["area"] <= 2000]["price"].mean()

#%%
from sklearn.linear_model import LinearRegression
#%%% 
# For Analysis of the Bathrooms, Bedrooms, and parking
def funcJustDoSimple(x : np.array, y :np.array, title : str):
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=32)

    simpleLinReg: SimpleLinearRegression = SimpleLinearRegression()
    simpleLinReg.fit(x_train, y_train)

    LinRegSklearn = LinearRegression()
    LinRegSklearn.fit(x_train.reshape(-1,1),y_train)
    y_linReg = LinRegSklearn.predict(x_test.reshape(-1,1))
    y_preds  : np.array =  simpleLinReg.predict(x_test)
    print(f"###########################  {title}  ############################################")
    print(f"Intercept : {simpleLinReg.b0:.2f}  Gradient {simpleLinReg.b1:.2f}")
    print(np.sqrt(mean_squared_error(y_test,y_preds)))
    
    print(f"Intercept : {LinRegSklearn.intercept_:.2f}  Gradient {LinRegSklearn.coef_[0]:.2f}")
    print(np.sqrt(mean_squared_error(y_test,y_preds)))
    print(f"###########################  {title}  ############################################")
    print()
    
    

x_bathrooms : np.array = df["bathrooms"].values
x_bedrooms : np.array = df["bedrooms"].values


funcJustDoSimple(x_bathrooms, df["price"], "BathRooms")
funcJustDoSimple(x_bedrooms, df["price"], "Bedrooms")
#%%w