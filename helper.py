import numpy as np


def embed_query(query, it):
    embedding = np.zeros((365,), dtype=int)
    
    for q in query:
        idx = np.where(q == it)
        embedding[idx] = 1

    return embedding