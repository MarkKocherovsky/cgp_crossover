import numpy as np
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
from effProg import effProg, cgp_active_nodes

def find_similarity(child, parent, c_out=[], p_out=[], mode='cgp', method='distance'):
    if method == 'proportion':
        return find_similarity_prop(child, parent, c_out, p_out, mode)
    elif method == 'distance':
        return find_similarity_distance(child, parent, c_out, p_out, mode)

def get_similarity_score(c, p):
    if c.size == 0 or p.size == 0:
        return 0

    all_active = np.concatenate((c, p), axis=0)
    all_unique = {tuple(x) for x in all_active}
    mapping = {v: str(k) for k, v in enumerate(all_unique)}

    c_str = [mapping[tuple(node)] for node in c]
    p_str = [mapping[tuple(node)] for node in p]
    
    v = Vocabulary()
    c_enc = v.encodeSequence(Sequence(c_str))
    p_enc = v.encodeSequence(Sequence(p_str))
    
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, _ = aligner.align(c_enc, p_enc, backtrace=True)
    return score

def find_similarity_distance(child, parent, c_out=[], p_out=[], mode='cgp'):
    if mode == 'cgp':
        c_active = child[cgp_active_nodes(child, c_out, opt=3).astype(np.int32)]
        p_active = parent[cgp_active_nodes(parent, p_out, opt=3).astype(np.int32)]
    elif mode == 'lgp':
        c_active = effProg(4, child)
        p_active = effProg(4, parent)
    else:
        raise ValueError("Unsupported mode")

    return get_similarity_score(c_active, p_active)

def find_similarity_prop(child, parent, c_out=[], p_out=[], mode='cgp'):
    if mode == 'cgp':
        c_active = cgp_active_nodes(child, c_out, opt=1).astype(np.int32)
        p_active = cgp_active_nodes(parent, p_out, opt=1).astype(np.int32)
    elif mode == 'lgp':
        c_active = effProg(4, child)
        p_active = effProg(4, parent)
    else:
        raise ValueError("Unsupported mode")

    retention = np.intersect1d(c_active, p_active)
    try:
        return retention.shape[0] / p_active.shape[0]
    except ZeroDivisionError:
        return 0

