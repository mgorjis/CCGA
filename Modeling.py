import numpy as np
import random
import statsmodels.api as sm
from scipy import stats
#from scipy.stats import norm
import timeit
from Utils  import *
import statsmodels.api as sm
import matplotlib.pyplot as plt


def ModelIt(x, *u, T,model):

    if model=='K-means':
        if len(x)>0:
            Theta = np.mean(x)
            #print (Theta, np.sum((x-Theta)**2))
            return Theta, np.sum((x-Theta)**2)
        else:
            return random.random(), np.nan     
        
    if model=='Linear-Trend':
        if len(x)>0:  
            Theta = list(stats.linregress(u,x)[0:2])
            return Theta, np.sum((x-(Theta[0]*u + Theta[1]))**2)
        else:
            return np.random.rand(2), np.inf
        
    if model=='Univariate-Gaussian':
        if len(x)>0:
            Theta = list(stats.norm.fit(x))
            return Theta, -np.sum(stats.norm(Theta[0],Theta[1]).pdf(x.transpose()))#.transpose()
        else:
            return np.random.rand(2),np.inf      
        
    if model=='Gamma':
        if len(x)>0:
            Theta = list(stats.gamma.fit(x,floc=0))  #shape,loc,scale
            return Theta, -np.sum(stats.gamma(Theta[0],Theta[1],Theta[2]).pdf(x))
        else:
            return np.random.rand(3),np.inf  
        
    if model == 'Poisson':
        if len(x)>0:
            return np.mean(x) #, -np.sum(np.multiply(pmfs,RMF).sum(axis=0))
        else:
            return random.random()
        
    if model == 'Poisson-GLM':
        try:
            u=u[0]
            u_ = sm.add_constant(u, prepend=False)
            model_ = sm.GLM(x, u_, family=sm.families.Poisson(link = sm.families.links.identity)).fit() 
            return model_.params, -model_.llf
        except:
            return np.array([ 0.02390766,  1.40261013]),np.inf  #u.shape[1]+1  np.random.rand(3)
                  
   

        
######################################################################

def SolveTheta(ParentTimes,ParentRegimes,x,u,K,C,T,model):
   
    RMF=np.zeros([K,T])
    ParentTimes=np.hstack([0,ParentTimes,T])
    for c in range(0,C+1):
        RMF[ParentRegimes[c],np.arange(int(ParentTimes[c]),int(ParentTimes[c+1]))]= 1 
        
    if model=='K-means':
        output =np.array([ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)])
        Error=np.nansum(output[:,1])
        
    if model=='Linear-Trend':
        output= np.array([ModelIt(x[0,RMF[k]==1],np.arange(0,T)[RMF[k]==1],T=T,model=model) for k in range(0,K)])
        Error = np.sum(output[:,1])
        
    if model=='Univariate-Gaussian':
        output= np.array([ModelIt(x[0,RMF[k]==1],np.arange(0,T)[RMF[k]==1],T=T,model=model) for k in range(0,K)])
        Error = np.sum(output[:,1])
            
    if model=='Gamma':
         output= np.array([ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)])   
         Error = np.sum(output[:,1])   
            
    if model=='Poisson':
        Theta=np.array([ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)])
        pmfs=stats.poisson(Theta[:]).pmf(x.transpose()).transpose()
        Error=-np.sum(np.multiply(pmfs,RMF).sum(axis=0))
        
    if model == 'Poisson-GLM':
        x=x.flatten()  #x[0]
        Theta= np.array([ModelIt(x[RMF[k]==1], u[RMF[k]==1,:],T=T,model=model) for k in range(0,K)])
        Error = np.sum(Theta[:,1])
        
        
     
    #if len(set(ParentRegimes))!=K:
        #Error = np.sign(Error)*Error*np.inf
    return Error


######################################################################



def Finalize(IndividualTimes,IndividualRegimes, x,  u , K,C,T, model, ): 
    
    RMF=np.zeros([K,T])
    IndividualTimes=np.hstack([0,IndividualTimes,T])
    for c in range(0,C+1):
        RMF[IndividualRegimes[c],np.arange(int(IndividualTimes[c]),int(IndividualTimes[c+1]))]= 1
            
    if model=='K-means':
        output =np.array([ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)])
        Theta = output[:,0]
        xhat = np.multiply(np.matrix(output[:,0]).transpose(),RMF).sum(axis=0)    
        plt.plot(x.transpose(),'g',label = r'$x(t)$')
        plt.plot(xhat.transpose(),'r', label = r'$\hat{x}(t)$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if model== 'Linear-Trend':
        output= np.array([ModelIt(x[0,RMF[k]==1],np.arange(0,T)[RMF[k]==1],T=T,model=model) for k in range(0,K)])
        Theta = np.vstack(output[:,0])
        a= np.matrix(Theta[:,0]).transpose() * np.matrix(np.arange(0,T)) + np.matrix(Theta[:,1]).transpose()  
        xhat=np.multiply(a,RMF).sum(axis=0)
        plt.plot(x.transpose(),'g',label = r'$x(t)$')
        plt.plot(xhat.transpose(),'r', label = r'$\hat{x}(t)$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    if model=='Univariate-Gaussian':
        output= np.array([ModelIt(x[0,RMF[k]==1],np.arange(0,T)[RMF[k]==1],T=T,model=model) for k in range(0,K)])
        Theta = np.vstack(output[:,0])
        muhat = (np.matrix(Theta[:,0])*RMF).sum(axis=0)
        mu_plus_sigma_hat =muhat+ (np.matrix(Theta[:,1])*RMF).sum(axis=0)
        mu_minus_sigma_hat =muhat- (np.matrix(Theta[:,1])*RMF).sum(axis=0)
        plt.plot(x.transpose(),'g',label = r'$x(t)$')
        plt.plot(muhat.transpose(),'r', label = r'$\hat{\mu}(t)$')
        plt.plot(mu_plus_sigma_hat.transpose(),'b', label = r'$\hat{\mu}(t)+\hat{\sigma}(t)$')
        plt.plot(mu_minus_sigma_hat.transpose(),'b', label = r'$\hat{\mu}(t)-\hat{\sigma}(t)$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
          
    if model=='Gamma':
        output= np.array([ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)]) 
        Theta = np.vstack(output[:,0])
        muhat = (np.multiply(np.matrix(Theta[:,0])  , np.matrix(Theta[:,2]) )*RMF).sum(axis=0)
        plt.plot(x.transpose(),'g',label = r'$x(t)$')
        plt.plot(muhat.transpose(),'r', label = r'$\hat{\mu}(t)$')
        
    if model=='Poisson':
        Theta=np.array([ModelIt(x[0,RMF[k]==1],T=T,model=model) for k in range(0,K)])
        muhat = (np.matrix(Theta[:])  * RMF)  #.sum(axis=0)
        plt.plot(x.transpose(),'g',label = r'$x(t)$')
        plt.plot(muhat.transpose(),'r', label = r'$\hat{\mu}(t)$')
        
    if model == 'Poisson-GLM':  
        x=x.flatten()  #x[0]  
        Theta= np.array([ModelIt(x[RMF[k]==1], u[RMF[k]==1,:],T=T,model=model) for k in range(0,K)])
        u_ = sm.add_constant(u, prepend=False)
        a = np.matrix(list(Theta[:,0]))*u_.transpose()
        muhat=np.multiply(a,RMF).sum(axis=0)
        plt.plot(x.transpose(),'g',label = r'$x(t)$')
        plt.plot(muhat.transpose(),'r', label = r'$\hat{\mu}(t)$')
        
    plt.figure(figsize=(16,9))    
    return Theta

######################################################################           
    


def GACD(x, *u, K, C, PopSize, NumGen, model):
    
    try:
        u=u[0]
    except:
        u=0
    
    T=x.shape[1]
    
    start = timeit.default_timer()

    Population=[CreatePopTimes(C,T) + CreatePopRegimes(K,C,T) for k in range(0,PopSize)]
    #Population[0] = [70,200,250,0,1,0,2]
    Cost=np.empty([PopSize])
    for i in range(0,PopSize):
        Cost[i]=SolveTheta(Population[i][0:C],Population[i][C:2*C+1],x, u,K=K,C=C,T=T, model=model)

    
    
    MaxFit = np.empty(NumGen)
    MinFit = np.empty(NumGen)
    Uniques = np.empty(NumGen)

    for gen in range(0,NumGen):
        #print(gen)
        #print([i[C:2*C+1] for i in Population])

        Uniques[gen]  = len([x for i, x in enumerate(Population) if x not in Population[0:i]])

        if Uniques[gen]==1:# <.1*PopSize:
            break

        index1= random.choice(range(0,PopSize))
        index2= random.choice(range(0,PopSize))
        Parent1= Population[index1]
        Parent2= Population[index2]
        Cost_Parent1 = Cost[index1]
        Cost_Parent2 = Cost[index2]

        # Crossover
        if random.random() > .5:    #crossover on times
            Ch1, Ch2 = [list(a) for a in CrossoverTimes(np.array(Parent1[0:C]),np.array(Parent2[0:C]))]
            Child = Ch1+Parent1[C:2*C+1]
        else:
            Ch1, Ch2  = [list(a) for a in CrossoverRegimes(Parent1[C:2*C+1],Parent2[C:2*C+1],K,C,T)]
            Child = Parent1[0:C] + Ch1

        # Mutation
        if random.random() > .5:
            Ch1 = MutationTimes(Child[0:C],K,C,T)     
            Child = Ch1+Parent1[C:2*C+1]   
        else:   
            Ch1 = MutationRegimes(Child[C:2*C+1],K,C)     
            Child = Parent1[0:C] + Ch1 

        # Child Fitness 
        Cost_Child=SolveTheta(Child[0:C],Child[C:2*C+1],x, u,K=K,C=C,T=T, model=model)

        #considers the individuals in the population with poorer fitness values than the offspring
        indices = np.where(Cost_Child<Cost)[0]
        contribution_diversity_worse = [contribution_diversity(Population[i],i,Population,C, T,alpha=1) for i in indices]
        counter = len(contribution_diversity_worse)

        if counter==0:   # replace worse
            index=np.argmax(Cost)
            Population[index]=Child;
            Cost[index]=Cost_Child

        #and finds the onewith the lowest contribution of diversity, cmin.
        if counter!=0:
            index_cmin=np.argmin(contribution_diversity_worse)
            cmin_contribution_diversity=np.min(contribution_diversity_worse)
            #Then, it compares the contribution of diversity of this element with the contribution 
            #of diversity of the offspring to the population (removing cmin from it).
            contribution_diversity_child = contribution_diversity(Child,indices[index_cmin],Population,C,T, alpha =1)

            if contribution_diversity_child>cmin_contribution_diversity:
                Population[indices[index_cmin]]=Child;
                Cost[indices[index_cmin]]=Cost_Child
            else: #replace worse 
                index=np.argmax(Cost)
                Population[index]=Child;
                Cost[index]=Cost_Child

        MinFit[gen]= np.min(Cost)
        MaxFit[gen]= np.max(Cost)

    stop = timeit.default_timer() 

    MinFit = MinFit[0:gen]
    Best_individual = Population[Cost.argsort()[1:][0]]
    Theta= Finalize(Best_individual[0:C],Best_individual[C:2*C+1], x, u,K=K, C=C, T=T, model=model)


    print("The GACD algorithm converged in " + str(np.floor(stop - start )) + ' seconds and in '+ str( gen+1) + ' generations')
    print("The best switch times are " + str([int(i) for i in Best_individual[0:C]]) + ' and the best regime order is '+ str(Best_individual[C:2*C+1]))  
    print("The minimum reached cost function is " + str(min(Cost)) )
    
    return Theta, MinFit

