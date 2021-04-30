# Regular GA

# Intialization
#a = map(lambda p: SolveTheta(p[0:C],p[C:2*C+1],x[0],K=K,C=C,T=T, model='K-means')[1],Population)b=list(a)


Population=[CreatePopTimes(C,T) + CreatePopRegimes(K,C,T) for k in range(0,Popsize)]
Cost=np.empty([Popsize])
for i in range(0,Popsize):
    Theta,  Cost[i]=SolveTheta(Population[i][0:C],Population[i][C:2*C+1],x[0],K=K,C=C,T=T, model=model)
    
    
    
    

MinFit = np.empty(NumGen)
Uniques = np.empty(NumGen)

for gen in range(0,NumGen):
    
    Uniques[gen]  = len([x for i, x in enumerate(Population) if x not in Population[0:i]])
    
    index1= random.choice(range(0,Popsize))
    index2= random.choice(range(0,Popsize))
    Parent1= Population[index1]
    Parent2= Population[index2]
    
    # Crossover
    
    if random.random() > .5:
        #crossover on times
        Ch1, Ch2 = [list([int(i) for i in a]) for a in CrossoverTimes(np.array(Parent1[0:C]),np.array(Parent2[0:C]))]
        Child1 = Ch1+Parent1[C:2*C+1]
        Child2 = Ch2+Parent2[C:2*C+1]
    else:
        Ch1, Ch2  = [list(a) for a in CrossoverRegimes(Parent1[C:2*C+1],Parent2[C:2*C+1],K)]
        Child1 = Parent1[0:C] + Ch1
        Child2 = Parent2[0:C] + Ch2
        
    # Mutation
    
    if random.random() > .5:
        Ch1 = MutationTimes(Child1[0:C],K,C,T)  
        Ch2 = MutationTimes(Child2[0:C],K,C,T)
        
        Child1 = Ch1+Parent1[C:2*C+1]
        Child2 = Ch2+Parent2[C:2*C+1]
        
    else:
        
        Ch1 = MutationRegimes(Child1[C:2*C+1],K,C,)  
        Ch2 = MutationRegimes(Child2[C:2*C+1],K,C)
        
        Child1 = Parent1[0:C] + Ch1
        Child2 = Parent2[0:C] + Ch2
        
    
    # Child Fitness
        
        Theta, Cost_Child1=SolveTheta(Child1[0:C],Child1[C:2*C+1],x[0],K=K,C=C,T=T, model=model)
        Theta, Cost_Child2=SolveTheta(Child2[0:C],Child2[C:2*C+1],x[0],K=K,C=C,T=T, model=model)
        
    # Replacement
        
        Worst_Members = Cost.argsort()[-2:]
        
        Population [Worst_Members[0]] = Child1
        Cost [Worst_Members[0]] = Cost_Child1
        Population [Worst_Members[1]] = Child2
        Cost [Worst_Members[1]] = Cost_Child2
        
    MinFit[gen]= np.min(Cost)
    
    
plt.plot(MinFit)

#plt.plot(Uniques)


# CD GA
Popsize = 50
NumGen = 5000


Population=[CreatePopTimes(C,T) + CreatePopRegimes(K,C,T) for k in range(0,Popsize)]
Cost=np.empty([Popsize])
for i in range(0,Popsize):
    Theta,  Cost[i]=SolveTheta(Population[i][0:C],Population[i][C:2*C+1],x[0],K=K,C=C,T=T, model=model)
#print(Population)       
#print(Cost)   
MaxFit = np.empty(NumGen)
MinFit = np.empty(NumGen)
Uniques = np.empty(NumGen)

for gen in range(0,NumGen):
    
    Uniques[gen]  = len([x for i, x in enumerate(Population) if x not in Population[0:i]])
    
    index1= random.choice(range(0,Popsize))
    index2= random.choice(range(0,Popsize))
    #print (index1,index2)
    
    Parent1= Population[index1]
    Parent2= Population[index2]
    
    Cost_Parent1 = Cost[index1]
    Cost_Parent2 = Cost[index2]
    
    # Crossover
    if random.random() > .5:
        #crossover on times
        Ch1, Ch2 = [list([int(i) for i in a]) for a in CrossoverTimes(np.array(Parent1[0:C]),np.array(Parent2[0:C]))]
        Child1 = Ch1+Parent1[C:2*C+1]
        Child2 = Ch2+Parent2[C:2*C+1]
    else:
        Ch1, Ch2  = [list(a) for a in CrossoverRegimes(Parent1[C:2*C+1],Parent2[C:2*C+1],K)]
        Child1 = Parent1[0:C] + Ch1
        Child2 = Parent2[0:C] + Ch2
        
    # Mutation
    if random.random() > .5:
        Ch1 = MutationTimes(Child1[0:C],K,C,T)  
        Ch2 = MutationTimes(Child2[0:C],K,C,T)        
        Child1 = Ch1+Parent1[C:2*C+1]
        Child2 = Ch2+Parent2[C:2*C+1]    
    else:   
        Ch1 = MutationRegimes(Child1[C:2*C+1],K,C)  
        Ch2 = MutationRegimes(Child2[C:2*C+1],K,C)    
        Child1 = Parent1[0:C] + Ch1
        Child2 = Parent2[0:C] + Ch2     
    
    # Child Fitness 
    Theta, Cost_Child1=SolveTheta(Child1[0:C],Child1[C:2*C+1],x[0],K=K,C=C,T=T, model=model)
    Theta, Cost_Child2=SolveTheta(Child2[0:C],Child2[C:2*C+1],x[0],K=K,C=C,T=T, model=model)
        
    # Replacement
    
    
    #print(Parent1,Parent2,Child1,Child2)
    #print(Cost_Parent1,Cost_Parent2,Cost_Child1,Cost_Child2)
        
    dp1c1=Dissimilarity(Parent1,Child1,C, T, alpha =1)
    dp1c2=Dissimilarity(Parent1,Child2,C, T, alpha =1)
    dp2c1=Dissimilarity(Parent2,Child1,C, T, alpha =1)
    dp2c2=Dissimilarity(Parent2,Child2,C, T, alpha =1)
        
    #----------------------------------    
    if (dp1c1+dp2c2)<=(dp1c2+dp2c1):
        #print('a')
        if  Cost_Child1<=Cost_Parent1:
            Population[index1]=Child1
            Cost[index1]=Cost_Child1
            
        if  Cost_Child2<=Cost_Parent2:
            Population[index2]=Child2
            Cost[index2]=Cost_Child2

    if (dp1c1+dp2c2)>(dp1c2+dp2c1):
        #print(b)
        if  Cost_Child1<=Cost_Parent2:
            Population[index2]=Child1
            Cost[index2]=Cost_Child1

        if  Cost_Child2<=Cost_Parent1:
            Population[index1]=Child2
            Cost[index1]=Cost_Child2
 
        
    MinFit[gen]= np.min(Cost)
    MaxFit[gen]= np.max(Cost)
    
       
        
 
plt.plot(MinFit,'g')
plt.plot(MaxFit,'r') 

print(Population[Cost.argsort()[1:][0]])
#plt.plot(Uniques)














