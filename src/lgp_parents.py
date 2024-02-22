import numpy as np
from numpy import random

class lgpParentGenerator:
	def __init__(self, max_p, max_r, max_d, bank, inputs = 1, n_bias = 10, arity = 2):
		self.max_p = max_p #number of parents
		self.max_r = max_r #number of instructions
		self.inputs = inputs #number of inputs
		self.n_bias = n_bias #number of constants
		self.max_d = max_d #number of destinations
		self.arity = arity #number of sources per instructions
		self.bank = bank #operators
		self.destinations, self.sources = self.get_registers()

	def generate_instruction(self):
		instruction = np.zeros(((2+self.arity),), dtype=np.int32)
		instruction[0] = random.choice(self.destinations) #destination index
		instruction[1] = random.randint(0, len(self.bank))
		instruction[2:] = random.choice(self.sources, (self.arity,))
		return instruction
	
	def get_registers(self):
		possible_indices = np.arange(0, self.max_d+self.n_bias+self.inputs+1)
		possible_destinations = np.delete(possible_indices, np.arange(1, self.n_bias+self.inputs+1)) #removes inputs for destination
		possible_sources = possible_indices
		return possible_destinations, possible_sources
		

	def generate_ind(self):
		#instruction_count = 5
		instruction_count = random.randint(2, self.max_r)
		#print(instruction_count)
		instructions = np.zeros((instruction_count, 2+self.arity))
		for i in range(instruction_count):
			instructions[i, :] = self.generate_instruction()
		#print(instructions)
		return instructions
		
	def generate_single_ind(self):
		#instruction_count = 5
		instruction_count = 1
		#print(instruction_count)
		instructions = np.zeros((instruction_count, 2+self.arity))
		for i in range(instruction_count):
			instructions[i, :] = self.generate_instruction()
		#print(instructions)
		return instructions

	def __call__(self):
		parents = []
		if self.max_p < 2:
			return(self.generate_ind())
		for i in range(0, self.max_p):
			parents.append(self.generate_ind())
		return parents
