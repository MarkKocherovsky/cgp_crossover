#Ragusa, V. R. (2023). Harnessing the Complexity of Natural Evolution to Enhance Evolutionary Algorithms. Michigan State University.
from scipy.stats import binom
import matplotlib.pyplot as plt
from random import randint
from collections import Counter


#return a list containing the probability mass function
# N1 and N2 are the population size before and after reproduction (parents and children)
# P is the number of parents required to make a single offspring (P=1 asexual, P=2 crossover, etc.)
def getWrightFisherDriftDist(N1,N2,P):
    p = 1/N1 #probability to select a parent, assuming neutral drift
    s = P*N2 #number of selection events, (parents per child * number of children)
    return [binom.pmf(n,s,p) for n in range(s+1)] #0 <= n <= s


#show the drift distribution for a population that starts at size 10, grows to size 35, and has 2 parents per offspring
N1 = 50
N2 = 60
numParents = 2
driftDist = getWrightFisherDriftDist(N1,N2,numParents)
plt.title("The drift distribution for\nN1:{}, N2:{}, P:{}".format(N1,N2,numParents))
plt.xlabel("Number of offspring this generation")
plt.ylabel("Fraction of parents expected to contribute\nto X offspring this generation")
plt.plot(driftDist,label="theory")
plt.legend()
plt.show()


#empirically verify drift distribution
population = [0]*N1 #init a population, no offspring recorded
for offspring in range(N2): #make offpsring
    for parent in range(numParents): #choose parents for each offspring
        parentID = randint(0,N1-1) #randomly select a parent (simulating neutral drift)
        population[parentID] += 1 #mark each parent as having contributed to an offspring


#compute offspring distribution from offspring counts
offspringCountFrequency = Counter(population) #how many parents had how many offspring?
X = []
Y = []
for numOffspring in sorted(offspringCountFrequency.keys()):
    X.append(numOffspring)
    Y.append(offspringCountFrequency[numOffspring]/N1) #convert count to density


#show the offspring distribution next to the theoretical distribution
plt.title("Empirical verification of the drift distribution for\nN1:{}, N2:{}, P:{}".format(N1,N2,numParents))
plt.xlabel("Number of offspring this generation")
plt.ylabel("Fraction of parents expected to contribute\nto X offspring this generation")
plt.plot(X,Y,marker="o",linestyle="",label="empirical Data")
plt.plot(driftDist,label="theory")
plt.legend()
plt.show()


#measure selection impact

#compute the EMD between two discrete distributions. We assume the dist.s share the same support.
# i.e., each P and Q contain a value for all x in the domain of P and Q. 
#https://en.wikipedia.org/wiki/Earth_mover%27s_distance#Computing_the_EMD
def earthMoverDistance(P,Q):
    N = len(P)
    assert N == len(Q) #the support of both distributions must be the same
    R = [0]
    for i in range(N):
        R.append(P[i]+R[i]-Q[i]) #for each bin, track how much work is needed
    return sum(map(abs,R)) #return the total work for all bins


#make sure the empirical distribution has the same support as the theoretical distribution
# i.e., scan for values in X that are implicitly zero and make them explicitly zero. 
empDist = [Y[X.index(i)] if i in X else 0 for i in range(numParents*N2+1)]


#compute the EMD between the theory and the empirical distribution
print("The selection impact is",earthMoverDistance(driftDist,empDist))







#compare to a population experiencing selection

#simple tournament selection algorithm. takes as input a vector of fitness scores
# returns the index of the selected individual
def tournamentSelect(fitnessVec,tournamentSize):
    winner = randint(0,len(fitnessVec)-1)
    for pick in range(tournamentSize-1):
        candidate = randint(0,len(fitnessVec)-1)
        if fitnessVec[candidate] > fitnessVec[winner]:
            winner = candidate
    return winner

population = [0]*N1 #init a population, no offspring recorded
fitnessVec = [randint(1,5) for _ in range(N1)] #generate some fitness scores
for offspring in range(N2): #make offpsring
    for parent in range(numParents): #choose parents for each offspring
        parentID = tournamentSelect(fitnessVec,5) #choose a parent fitness-proportionally 
        population[parentID] += 1 #mark each parent as having contributed to an offspring

#compute offspring distribution from offspring counts (same as above)
offspringCountFrequency = Counter(population) #how many parents had how many offspring?
X = []
Y = []
for numOffspring in sorted(offspringCountFrequency.keys()):
    X.append(numOffspring)
    Y.append(offspringCountFrequency[numOffspring]/N1) #convert count to density


#show the offspring distribution next to the theoretical distribution (same as above)
plt.title("The selection distribution for\nN1:{}, N2:{}, P:{}".format(N1,N2,numParents))
plt.xlabel("Number of offspring this generation")
plt.ylabel("Fraction of parents expected to contribute\nto X offspring this generation")
plt.plot(X,Y,marker="o",linestyle="",label="empirical Data")
plt.plot(driftDist,label="theory")
plt.legend()
plt.show()

#make sure the empirical distribution has the same support as the theoretical distribution (same as above)
# i.e., scan for values in X that are implicitly zero and make them explicitly zero. 
empDist = [Y[X.index(i)] if i in X else 0 for i in range(numParents*N2+1)]


#compute the EMD between the theory and the empirical distribution (same as above)
print("The selection impact is",earthMoverDistance(driftDist,empDist))
