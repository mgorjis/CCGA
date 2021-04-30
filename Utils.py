import numpy as np
from collections import OrderedDict
import random

def CreatePopRegimes(K,C,T):
    Regimes=np.arange(0,K)
    IndividualClusters=np.zeros([C+1])
    IndividualClusters=list(map(int,IndividualClusters))
    c=0
    while c<C+1:
        x=random.choice(Regimes)
        IndividualClusters[c]=x
        Regimes=np.arange(K)
        Regimes=np.delete(Regimes,x)
        c=c+1
    
    IndividualClusters = sortgen(IndividualClusters,K,C,T)
    return IndividualClusters

    ################################################

def CreatePopTimes(C,T):
    #times=np.arange(0,T)
    #IndividualTimes=np.sort([random.choice(times) for c in range(0,C)   ])
    IndividualTimes = np.sort(random.sample(range(1,T),C))
    return list(IndividualTimes)

     ################################################

def sortgen(a,K,C,T):
    b=a
    if list(set(a))==list(range(0,K)):
        unique=list(OrderedDict.fromkeys(a))
        b=np.empty(C+1)
        for i in range(0,C+1):
            b[i]=unique.index(a[i])
    b=list(map(int,b))
    return b

      ################################################


def CrossoverRegimes(Parent1,Parent2,K,C,T):

    if K>=3:
        list1=Parent1[2:-1]
        list2=Parent2[3:]
        pairwise = zip (list1, list2)
        matched_digits = [idx for idx, pair in enumerate(pairwise) if pair[0] != pair[1]]
        right=np.array(matched_digits)+2
        
        list1=Parent1[3:]
        list2=Parent2[2:-1]
        pairwise = zip (list1, list2)
        matched_digits = [idx for idx, pair in enumerate(pairwise) if pair[0] != pair[1]]
        left=np.array(matched_digits)+2
        
        possible_crosspoints= list( set(right) &  set(left) ) 
        if len(possible_crosspoints):
            Crosspoint=random.choice( possible_crosspoints )
            Child1=Parent1[0:Crosspoint+1] + Parent2[Crosspoint+1:]
            Child2=Parent2[0:Crosspoint+1] + Parent1[Crosspoint+1:]
            
            Child1 = sortgen(Child1,K,C,T)
            
        else:
            Child1=Parent1
            Child2=Parent2
    else:
        Child1=Parent1
        Child2=Parent2
    return Child1, Child2


    ##############################################

def CrossoverTimes(Parent1,Parent2):

    Lambda=random.random()
    Child1=  np.floor( Lambda*Parent1 + (1-Lambda)*Parent2 )
    Child2=  np.floor( (1-Lambda)*Parent1+(Lambda)*Parent2 )
    return np.sort(Child1), np.sort(Child2)

    ##############################################3

def MutationRegimes(Parent,K,C):

    if K>=4:
        mutate_site=random.choice( range(2,C+1) )
        possible_new_regime=range(0,K)
        if mutate_site==C :  
            possible_new_regime=np.delete(possible_new_regime,[Parent[mutate_site-1],Parent[mutate_site]])
        else:
            possible_new_regime=np.delete(possible_new_regime,[Parent[mutate_site-1],Parent[mutate_site],Parent[mutate_site+1]])
        
        Parent[mutate_site]=random.choice(possible_new_regime)
    return Parent

    ###############################################
def MutationTimes(Parent_,K,C,T):
    Parent = [1]+Parent_+[T]
    mutation_position = random.choice(range(0,C)) +1 
    MR = np.arange(Parent[mutation_position-1]+1, Parent[mutation_position+1]-1)
    try:
        Parent[mutation_position] = random.choice(MR)
    except:
        True
    return Parent[1:-1]    


###############################################

def DissimilarityTimes(Parent1,Parent2,C,T):
    return np.sum(np.abs(np.array(Parent1) - np.array(Parent2)))/((C*T-1))

###############################################

def DissimilarityRegimes(Parent1,Parent2,C):
    try:
        return sum([Parent1[i]!=Parent2[i] for i in range(0,C+1)])/(C-1)
    except: 
        return 0

###############################################

def Dissimilarity(Parent1,Parent2,C, T, alpha =1 ):
    a = DissimilarityTimes(Parent1[0:C],Parent2[0:C],C,T)
    b=  DissimilarityRegimes(Parent1[C:2*C+1],Parent2[C:2*C+1],C)
    return a + (alpha * b)

###############################################

def contribution_diversity(self,index_tobe_removed,Population, C,T,alpha=1):
    pop_i_removed=Population[:]  #deep copy
    del pop_i_removed[index_tobe_removed]
    cd = [Dissimilarity(self,i,C,T,alpha) for i in pop_i_removed]
    return np.min(cd)
    
  
    
    
    


#def Convert2RMF(ParentTimes,ParentRegimes,K,C,T):
    #RMF=np.zeros([K,T])
    #ParentTimes=np.hstack([0,ParentTimes,T])
    #for c in range(0,C+1):
        #RMF[ParentRegimes[c],range(ParentTimes[c],ParentTimes[c+1])]= 1
    #return RMF


#####################################3



#def MutationTimes(Parent,K,C,T):
    #import random
    #mutation_position = random.choice(range(0,C))
    
    #if mutation_position == 0:
        #MR = range(1, Parent[1]-1)
    #elif mutation_position == C-1:
        #MR = range(Parent[mutation_position-1]+1 ,T) 
    #else:
        #MR = range(Parent[mutation_position-1]+1, Parent[mutation_position+1]-1)
    #try:
        #Parent[mutation_position] = random.choice(MR) 
    #except:
        #True
    #return Parent

