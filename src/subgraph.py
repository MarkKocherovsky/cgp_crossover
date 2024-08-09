import numpy as np
from numpy import random


#Implementation of SubGraph Crossover from Kalkreuth's work
#Mark Kocherovsky
#Feb 2024

#Kalkreuth 2017
#n_i: Number of inputs
#m: Upper node number limit, null by default
#I List of input nodes, null by default
#N_F List of number of body nodes [x, x+1,...,x_n]
def RandomNodeNumber(n_i=1, I=None, N_f=None, m=None):
    if N_f is None:
        N_f = []
    if I is None:
        I = []
    N_R = []  #Initialize an Empty List to store random input and function node numbers
    if len(N_f) > 0:  #check if function node numbers have been passed as argument
        if m is not None:  #check if a node number limit has been passed to the function
            #determine a sublist of N_F where the list elements X of N_F are less or equal to m
            N_m = list(N_f[N_f <= m])
            if len(N_m) == 0:  #if the sublist is empty, there are no function nodes before m
                i = random.randint(0, n_i)  #generate a random input node
                N_R.append(i)  #append the random input to the list
            else:
                if len(N_m) - 1 <= 0:
                    i = random.randint(0, 1)
                else:
                    i = random.randint(0,
                                       len(N_m) - 1)  # generate a random integer in the range from 0 to |N_m|-1
                # inclusive(?)
                N_R.append(N_m[i])  #use i as index and get the node number from N_F
        else:  #otherwise, randomly select a node number in the range from 0 to |N_F|-1 inclusive
            i = random.randint(0, len(list(N_f)) - 1)
            N_R.append(N_f[i])
    if len(I) > 0:
        #Select a random input node number in the range from 0 to |I|-1 inclusive by chance
        if len(I) - 1 <= 0:
            j = 0
        else:
            j = random.randint(0, len(I) - 1)
        N_R.append(j)
    #select one node number from the list N_R by chance
    try:
        r = random.randint(0, len(N_R) - 1)
    except ValueError:
        r = 0
    return N_R[r]


#Kalkreuth 2017
#P1: Genome of first parent
#P2: Genome of second parent
#M1: List of active nodes of the first parent
#M2: List of active nodes of the second parent
def DetermineCrossoverPoint(P1, P2, M1, M2):
    if len(M1) > 0 and len(M2) > 0:  #The opposite shouldn't happen but just in case!
        if len(M1) > 0:
            a = np.min(M1)  #Determine minimum node number of M1
            b = np.max(M1)  #Determine maximum node number of M1
            if a == b:
                CP1 = a
            else:
                CP1 = random.randint(a, b)  #Choose the first possible crossover point by chance
        else:
            CP1 = np.PINF
        if len(M2) > 0:
            a = np.min(M2)  #Determine minimum node number of M2
            b = np.max(M2)  #Determinem maximum node number of M2
            if a == b:
                CP2 = a
            else:
                CP2 = random.randint(a, b)
        else:
            CP2 = np.PINF
        return int(np.min([CP1, CP2]))  #The crossover point is the minimum of the possible points
    else:
        return -1


def DetermineActiveNodes(individual, first_body_node=11, arity=2):
    active_nodes = []
    ind = individual[0]
    output_nodes = individual[1]

    def get_body_node(n_node):
        node = ind[n_node - first_body_node]
        for a in range(arity):
            prev_node = node[a]
            if prev_node > first_body_node:  #inputs
                if prev_node not in active_nodes:
                    active_nodes.append(prev_node)
                get_body_node(prev_node)

    for o in range(len(output_nodes)):
        node = output_nodes[o]
        if node >= first_body_node:
            get_body_node(node)
            if node not in active_nodes:
                active_nodes.append(node)
        else:
            if node not in active_nodes:
                active_nodes.append(node)
    return active_nodes


#Kalkreuth 2017
#P1 Genome of the first parent
#P2 Genome of the second parent
#n_i Number of Inputs
def SubgraphCrossover(P1, P2, max_n, n_i=1, first_body_node=11, arity=2):
    density_distro = np.zeros(max_n*3)
    G1 = P1[0].copy()  #store the genome of parent p1 in g1
    G2 = P2[0].copy()  #store the genome of parent p2 in g2
    O1 = P1[1].copy()
    O2 = P2[1].copy()

    M1 = DetermineActiveNodes([G1, O1])
    M2 = DetermineActiveNodes([G2, O2])
    #print(M1, M2)
    n_g = G1.shape[0]  #Determine Number of Genes

    C_P = DetermineCrossoverPoint(G1, G2, M1, M2)  #Determine Crossover Point
    if C_P < 0:  #if neither have active nodes
        print("No Active Nodes?")
        print(G1, O1)
        print(G2, O2)
        return (G1, O1)
    p_c = C_P + 1 - first_body_node
    density_distro[p_c] += 1
    G0 = np.concatenate((G1[:p_c], G2[p_c:]),
                        axis=0)  #Copy the parts before and after crossover from G1 and G2 respectively
    O = O2.copy()  #back of the list so self explanatory really
    #Create the list of active function nodes of the offspring
    #Determine and store a sublist of M1 where the list elements of M are less or equal to CP
    #print(C_P)
    #print(M1)
    M1 = np.array(M1)
    M2 = np.array(M2)
    NA1 = M1[M1 <= C_P]
    NA2 = M2[M2 > C_P]

    if NA1.shape[0] > 0 and NA2.shape[0] > 0:  #check if both lists contain active nodes
        #print(NA1[-1])
        #print(NA2[0])
        nF = NA1[-1]  #Determine first active node before CP
        nB = NA2[0]  #Determine the first active node after CP
        G0 = NeighborhoodConnect(nF, nB, G0, first_body_node, arity)

    NA = np.concatenate((NA1, NA2))  #combine lists
    if NA.shape[0] > 0:
        G0, O = RandomActiveConnect(n_i, NA, C_P, G0, O, first_body_node, arity)
    return (G0, O), density_distro


#Kalkreuth 2017
#nF: Number of the first active node before the crossover point
#nB: Number of the first active node behind the crossover point
#G0: Offspring Genome
def NeighborhoodConnect(nF, nB, G0, first_body_node=11, arity=2):
    #print('neighborhood connect')
    #print(nF)
    #print(nB)
    #print(G0)
    """
    if nB >= nF:
    	print(f'nB {nB} >= nF {nF}')
	if nF >= nB:
		print(f'{nF} >= {nB} - {first_body_node}')
	if nB >= G0.shape[0]:
		print(f'nF {nF}')
		print(f'nB {nB}')
		print(f'G0 {G0}')
	"""
    #print(f'G0[{nB-first_body_node}, 0] = {nF}')
    G0[nB - first_body_node, 0] = nF
    return G0


#Kalkreuth 2017
# n_i: number of inputs
# NA: List of active function nodes
# CP: The Crossover Point
# G0: Genome of the Offspring
# O: Output Nodes
def RandomActiveConnect(n_i, NA, CP, G0, O, first_body_node=11, arity=2):
    #print('random active connect')
    I = []  #get input nodes
    for n in G0:
        for a in range(0, arity):
            if n[a] < first_body_node and n[a] not in I:
                I.append(n[a])
    #print(f'I {I}')
    #print(f'Crossover Point {CP}')
    #print(f'First Body Node {first_body_node}')
    #print(f'NA {NA}')
    for n in NA:  #iterate over the active nodes
        #print(f'n {n}')
        if n > CP:  #if node is greater than xover point
            #print(f'\tn > CP: {n > CP}')
            node = G0[n - first_body_node]  #get connection genes
            #print(f'\tnode {n} = G0[{n-first_body_node}] = {node}')
            GC = node[:arity]
            #print(f'\tGC = {GC}')
            for i in range(arity):  #iterate over connection genes
                g = GC[i]
                #print(f'\t\tg = GC[{i}] = {g}')
                if g not in NA:  #if the current connection gene is not connected to an active function node
                    #print(f'\t\t\t{g} not in NA')
                    #print(f'\t\t\t{node}')
                    #print(f'\t\t\t{G0[node, i]}')
                    G0[n - first_body_node, i] = RandomNodeNumber(n_i, I, NA, CP)
                #print(f'\t\t\t{G0[node, i]}')
    for o in O:  #Adjust output genes
        if o not in NA:  #if output is connected to an inactive nodes
            o = RandomNodeNumber(I, NA)
    return G0, O
