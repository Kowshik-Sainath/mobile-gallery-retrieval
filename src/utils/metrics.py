"""
Evaluation Metrics: Recall@K and NDCG@K.
"""

import numpy as np

def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Computes Recall@K where ground truth for query i is index i.
    Args:
        similarity_matrix: (N_query, N_gallery) Cosine similarity matrix.
        k_values: list of integers.
    Returns:
        dict mapping k -> recall percentage.
    """
    num_queries = similarity_matrix.shape[0]
    rankings = np.argsort(-similarity_matrix, axis=1)

    recalls = {}
    for k in k_values:
        top_k = rankings[:, :k]
        hits = 0
        for i in range(num_queries):
            if i in top_k[i]:
                hits += 1
        recalls[f"R@{k}"] = (hits / num_queries) * 100.0

    return recalls


def compute_ndcg_at_k(similarity_matrix, k=10):
    """
    Computes NDCG@K where ground truth rank is 1 for exact match index i.
    """
    num_queries = similarity_matrix.shape[0]
    rankings = np.argsort(-similarity_matrix, axis=1)
    
    ndcg_list = []
    for i in range(num_queries):
        rank_pos = np.where(rankings[i, :k] == i)[0]
        if len(rank_pos) > 0:
            dcg = 1.0 / np.log2(rank_pos[0] + 2)
            ndcg_list.append(dcg)
        else:
            ndcg_list.append(0.0)

    return np.mean(ndcg_list) * 100.0
