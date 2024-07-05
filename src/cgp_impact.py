import numpy as np
#Ragusa, V. R. (2023). Harnessing the Complexity of Natural Evolution to Enhance Evolutionary Algorithms. Michigan State University.
from scipy.stats import binom
import matplotlib.pyplot as plt
from random import randint
from collections import Counter

from numpy import random

class SelectionImpact:
	# N1 and N2 are the population size before and after reproduction (parents and children)
	# P is the number of parents required to make a single offspring (P=1 asexual, P=2 crossover, etc.)
	def __init__(self, N1, N2, P):
		self.N1 = N1
		self.N2 = N2
		self.P = P
		self.drift_dist = self.getWrightFisherDriftDist()
	#return a list containing the probability mass function
	def getWrightFisherDriftDist(self):
		p = 1/self.N1 #probability to select a parent, assuming neutral drift
		s = self.P*self.N2 #number of selection events, (parents per child * number of children)
		return [binom.pmf(n,s,p) for n in range(s+1)] #0 <= n <= s

	#compute the EMD between two discrete distributions. We assume the dist.s share the same support.
	# i.e., each P and Q contain a value for all x in the domain of P and Q. 
	#https://en.wikipedia.org/wiki/Earth_mover%27s_distance#Computing_the_EMD
	def earthMoverDistance(self, P,Q):
		N = len(P)
		assert N == len(Q) #the support of both distributions must be the same
		R = [0]
		for i in range(N):
			R.append(P[i]+R[i]-Q[i]) #for each bin, track how much work is needed
		return sum(map(abs,R)) #return the total work for all bins

	def __call__(self, population):
		#compute offspring distribution from offspring counts (same as above)
		offspringCountFrequency = Counter(population) #how many parents had how many offspring?
		X = []
		Y = []
		for numOffspring in sorted(offspringCountFrequency.keys()):
		    X.append(numOffspring)
		    Y.append(offspringCountFrequency[numOffspring]/self.N1) #convert count to density
		empDist = [Y[X.index(i)] if i in X else 0 for i in range(self.P*self.N2+1)]
		return self.earthMoverDistance(self.drift_dist, empDist)

