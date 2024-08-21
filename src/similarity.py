from alignment.sequence import Sequence
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
from alignment.vocabulary import Vocabulary

from effProg import *
def find_similarity(child, parent, first_body_node, c_out=[], p_out=[], mode='cgp', method='distance'):
    return {
        'proportion': find_similarity_prop,
        'distance': find_similarity_distance
    }[method](child, parent, first_body_node, c_out, p_out, mode)


def get_similarity_score(c, p):
    try:
        all_active = np.concatenate((c, p), axis=0)
    except ValueError:
        return 0

    all_unique = {tuple(x) for x in all_active}
    mapping = {v: str(i) for i, v in enumerate(all_unique)}
    c_str = [mapping[tuple(node)] for node in c]
    p_str = [mapping[tuple(node)] for node in p]

    v = Vocabulary()
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, _ = aligner.align(v.encodeSequence(Sequence(c_str)), v.encodeSequence(Sequence(p_str)), backtrace=True)
    return score


def find_similarity_distance(child, parent, first_body_node, c_out=[], p_out=[], mode='cgp'):
    if mode == 'cgp':
        c_active = child[np.array(cgp_active_nodes(child, c_out, first_body_node, opt=3), dtype=np.int32)]
        p_active = parent[np.array(cgp_active_nodes(parent, p_out, first_body_node, opt=3), dtype=np.int32)]
    elif mode == 'lgp':
        c_active, p_active = effProg(4, child), effProg(4, parent)
    return get_similarity_score(c_active, p_active)


def find_similarity_prop(child, parent, c_out=[], p_out=[], mode='cgp'):
    if mode == 'cgp':
        c_active = np.array(cgp_active_nodes(child, c_out, first_body_node, opt=1), dtype=np.int32)
        p_active = np.array(cgp_active_nodes(parent, p_out, first_body_node, opt=1), dtype=np.int32)
        retention = np.intersect1d(c_active, p_active)
    elif mode == 'lgp':
        c_active, p_active = effProg(4, child), effProg(4, parent)
        retention = np.array(list(set(map(tuple, c_active)) & set(map(tuple, p_active))))
    try:
        return retention.shape[0] / p_active.shape[0]
    except ZeroDivisionError:
        return 0
