#%%
### ###################################################
#       Coded by : Rose (Pratama Azmi A)
#       Date : Unknown 
#       Text editor : Vscode + VIM
#       Static typing version 
##################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple


df : pd.DataFrame = pd.read_csv("home.txt", delimiter=",", names=["size","bedroom","price"])

print("Mean of dataFrame \n%s" %(df.mean()))
print("Std of dataframe \n%s" %(df.std()))


df : pd.DataFrame = (df - df.mean()) / df.std() # Basically make the scales COMPARABLE ( Nyamain Scale)

#Create Matrices and set Hyperparameters ( Buat data maksudnya)
X : np.ndarray = df.iloc[:,0:2] # 0 to n-1
ones : np.array = np.ones(shape=(X.shape[0], 1))
print("x Shape %s" %str(X.shape))
print("ones Shape %s" %str(ones.shape)) # Represent the bias / ( Intercept )

## We add this bias stuff / randomity / intercept whatever add THIS STUFF ( TO REPRESENT the Uncertainty )
X = np.concatenate([X,ones], axis=1)

y : np.ndarray = df.iloc[:, 2:3].values
print(y.shape)

## Set  hyperparams And perpare the loss also the GRADIENT DESCENT STUUFF
######################################################################################################################################################
######################################################################################################################################################
alpha : float = 0.001 # Basically the learning rate
iters : int = 1000 # Iteration for doing the GRADIENT DESCENT
theta : np.ndarray = np.zeros(shape=(1,3)) # FoR STORING GRADIENT
print(theta)
def computeCost(X : np.ndarray, y : np.ndarray, theta : np.ndarray ) -> float:
    ## Basically ini cuman mean squared error aja, but disini aku menggunakan MSE rather RMSE
    mse : np.ndarray = np.power( ((X @ theta.T) -y), 2) # do dot product then calculate the difference
    return np.sum(mse) / (2* len(X)) # itu dua untuk cancel turunan dari MSE

def gradientDescent(X : np.ndarray, y : np.array, theta : np.array, iters :int, alpha : float) -> Tuple[np.ndarray, float]:
    cost : np.ndarray = np.zeros(iters)
    
    # do the gradient descent for n iterations
    for i in range(iters):
        ## Update gradient bang
        theta = theta - (alpha/len(X)) * np.sum( X * (X @ theta.T - y),axis=0) 
        print(theta)
        cost[i] = computeCost(X,y,theta)
    
    return theta, cost
######################################################################################################################################################
######################################################################################################################################################
# %%

######################################################################################################################################################
######################################################################################################################################################

g, cost = gradientDescent(X,y,theta, iters, alpha)

print("final Cost %s" %(computeCost(X,y,g))) # The last iterations to 1000

######################################################################################################################################################
######################################################################################################################################################
# %%

fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
plt.show()
