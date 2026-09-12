"""
On-Device Duplicate & Near-Duplicate Suppression Module.
Adapted from idealo/imagededup (pHash + Cosine Embedding Clustering).
"""

import numpy as np

try:
    from imagededup.methods import PHash
    IMAGEDEDUP_AVAILABLE = True
except ImportError:
    IMAGEDEDUP_AVAILABLE = False


class GalleryDeduplicator:
    """
    On-device deduplication & cluster collapse module for phone photo galleries.
    """
    def __init__(self, max_hamming_distance=10, max_cosine_distance=0.15):
        self.max_hamming_distance = max_hamming_distance
        self.max_cosine_distance = max_cosine_distance
        
        if IMAGEDEDUP_AVAILABLE:
            self.phasher = PHash()
        else:
            self.phasher = None

    def compute_phash(self, image_dir_or_dict):
        """
        Computes pHash values using imagededup library if directory provided, or returns mock hashes.
        """
        if IMAGEDEDUP_AVAILABLE and isinstance(image_dir_or_dict, str):
            return self.phasher.encode_images(image_dir_or_dict)
        return {}

    def cluster_embeddings(self, embeddings, thresholds=None):
        """
        Clusters photo embeddings into near-duplicate groups using cosine distance.
        Args:
            embeddings: (N, D) normalized numpy array.
            thresholds: float cosine distance threshold.
        Returns:
            clusters: list of lists containing image indices in each cluster.
        """
        thresh = thresholds if thresholds is not None else self.max_cosine_distance
        num_images = embeddings.shape[0]
        
        # Calculate cosine similarity and distance
        sim_matrix = np.dot(embeddings, embeddings.T)
        dist_matrix = 1.0 - sim_matrix
        
        visited = set()
        clusters = []

        for i in range(num_images):
            if i in visited:
                continue
            
            # Find all images within threshold distance
            duplicate_indices = np.where(dist_matrix[i] <= thresh)[0].tolist()
            clusters.append(duplicate_indices)
            visited.update(duplicate_indices)

        return clusters

    def collapse_retrieval_duplicates(self, ranked_indices, clusters):
        """
        Collapses ranked search results by taking only the highest-ranked candidate per near-duplicate cluster.
        """
        # Map item index to cluster ID
        index_to_cluster = {}
        for cluster_id, cluster_members in enumerate(clusters):
            for member in cluster_members:
                index_to_cluster[member] = cluster_id

        seen_clusters = set()
        collapsed_results = []

        for idx in ranked_indices:
            cluster_id = index_to_cluster.get(idx, idx)
            if cluster_id not in seen_clusters:
                seen_clusters.add(cluster_id)
                collapsed_results.append(idx)

        return collapsed_results
