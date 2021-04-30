import numpy as np
import random

from Modeling import ModelIt
from Utils import *

def Finalize(IndividualTimes,IndividualRegimes, x, *u , K,C,T, model, ): 
    
    RMF=np.zeros([K,T])
    IndividualTimes=np.hstack([0,IndividualTimes,T])
    for c in range(0,C+1):
        RMF[IndividualRegimes[c],np.arange(int(IndividualTimes[c]),int(IndividualTimes[c+1]))]= 1
            
    if model=='K-means':
        Theta=[ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)]
        theta_t=np.zeros([1,T])
        for t in range(0,T):
            theta_t[0,t] = Theta[np.argmax(RMF[:,t])]
        plt.plot(x.transpose(),'g',label = 'x(t)')
        plt.plot(theta_t.transpose(),'r', label = 'xhat(t)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    if model== 'Linear-Trend':
        Theta= np.array([ModelIt(x[0,RMF[k]==1],np.arange(0,T)[RMF[k]==1],T=T,model=model) for k in range(0,K)])
        a= np.matrix(Theta[:,0]).transpose() * np.matrix(np.arange(0,T)) + np.matrix(Theta[:,1]).transpose() 
        xhat=np.multiply(a,RMF).sum(axis=0)
        plt.plot(x.transpose(),'g',label = 'x(t)')
        plt.plot(xhat.transpose(),'r', label = 'xhat(t)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
    return Theta

           
    
