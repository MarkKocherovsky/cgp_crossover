import numpy as np
import alignment
from effProg import *
def find_similarity(child, parent, c_out = [], p_out = [], mode = 'cgp', method = 'distance'): #get similarity between active nodes
	if method == 'proportion':
		return find_similarity_prop(child, parent, c_out, p_out, mode)
	elif method == 'distance':
		return find_similarity_distance(child, parent, c_out, p_out, mode)

def find_similarity_distance(child, parent, c_out = [], p_out = [], mode = 'cgp'):
	if mode == 'cgp':
		first_body_node = 11
		c_active = child[np.array(cgp_active_nodes(child, c_out, opt = 3)).astype(np.int32)]
		p_active = parent[np.array(cgp_active_nodes(parent, p_out, opt = 3)).astype(np.int32)]
		all_active = np.concatenate((c_active, p_active), axis = 0)
		all_unique = set([tuple(x) for x in all_active])
		keys = list(range(0, len(all_unique)))
		keys = map(str, keys)
		mapping = dict(map(lambda i,j : (i,j) , keys,all_unique)) #https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary
		mapping = {v: k for k, v in mapping.items()}
		c_str = [mapping[tuple(node)] for node in c_active]
		p_str = [mapping[tuple(node)] for node in p_active]
		
		from alignment.sequence import Sequence
		from alignment.vocabulary import Vocabulary
		from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
		v = Vocabulary()
		c_enc = v.encodeSequence(Sequence(c_str))
		p_enc = v.encodeSequence(Sequence(p_str))
		scoring = SimpleScoring(2, -1)
		aligner = GlobalSequenceAligner(scoring, -2)
		score, encodeds = aligner.align(c_enc, p_enc, backtrace=True)
		return score
def find_similarity_prop(child, parent, c_out = [], p_out = [], mode = 'cgp'): #get proportion of conservative active genes
	
	if mode == 'cgp':
		first_body_node = 11
		c_active = np.array(cgp_active_nodes(child, c_out, opt = 1)).astype(np.int32)
		p_active = np.array(cgp_active_nodes(parent, p_out, opt = 1)).astype(np.int32)
		
		retention = np.intersect1d(c_active, p_active)
		return retention.shape[0]/p_active.shape[0]
	elif mode == 'lgp':
		first_body_node = 12
		#print(parent)
		c_eff = effProg(4, child)
		p_eff = effProg(4, parent)
		#print(p_eff)
		#https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
		c_set = set([tuple(x) for x in c_eff])
		p_set = set([tuple(x) for x in p_eff])
		retention = np.array([x for x in c_set & p_set])
		try:
			return retention.shape[0]/p_eff.shape[0]
		except ZeroDivisionError:
			return 0
