import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import partial
import logging
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusterStats:
    size: int
    centroid: np.ndarray
    variance: float

def compute_cluster_stats(X: np.ndarray, labels: np.ndarray) -> Dict[int, ClusterStats]:
    """Compute statistical properties for each cluster"""
    stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            variance = np.var(cluster_points, axis=0).mean()
            stats[label] = ClusterStats(size=np.sum(mask), centroid=centroid, variance=variance)
        else:
            stats[label] = ClusterStats(size=0, centroid=np.zeros(X.shape[1]), variance=0.0)
    return stats

def find_merge_target_connectivity(small_cluster_idx, X, labels, small_clusters, connectivity):
    """Find the best cluster to merge a small cluster into using connectivity constraints"""
    mask_small = labels == small_cluster_idx
    points_small = X[mask_small]
    indices_small = np.where(mask_small)[0]
    
    if len(points_small) == 0:
        logger.warning(f"Attempting to merge empty cluster {small_cluster_idx}")
        return small_cluster_idx, None
    
    cluster_stats = compute_cluster_stats(X, labels)
    
    # Build graph from connectivity
    if connectivity is not None:
        try:
            graph = nx.from_numpy_array(connectivity)
            
            # Find connected clusters
            connected_clusters = set()
            for idx in indices_small:
                neighbors = list(graph.neighbors(idx))
                neighbor_labels = set(labels[neighbors])
                connected_clusters.update(neighbor_labels)
            
            connected_clusters.discard(small_cluster_idx)
            connected_clusters = connected_clusters - small_clusters
            
            if connected_clusters:
                # Choose the cluster with strongest connectivity
                max_weight = -float('inf')
                merge_target = None
                
                for connected_cluster in connected_clusters:
                    mask_other = labels == connected_cluster
                    indices_other = np.where(mask_other)[0]
                    
                    # Sum of edge weights between clusters
                    weight_sum = sum(graph[i][j]['weight'] for i in indices_small
                                     for j in indices_other if graph.has_edge(i, j))
                    
                    if weight_sum > max_weight:
                        max_weight = weight_sum
                        merge_target = connected_cluster
                
                if merge_target is not None:
                    logger.info(f"Merging cluster {small_cluster_idx} into {merge_target} based on connectivity")
                    return small_cluster_idx, merge_target
        except Exception as e:
            logger.warning(f"Error in connectivity-based merging: {e}. Falling back to variance-based merging.")
    
    # Fallback to Ward linkage (variance-based)
    logger.info(f"Using Ward linkage fallback for cluster {small_cluster_idx}")
    min_variance_increase = float('inf')
    merge_target = None
    
    # Get all candidate clusters (excluding small clusters)
    candidate_clusters = set(np.unique(labels)) - small_clusters
    
    for other_label in candidate_clusters:
        mask_other = labels == other_label
        points_other = X[mask_other]
        
        if len(points_other) == 0:
            continue  # Skip empty clusters
        
        # Compute combined variance
        combined_points = np.vstack([points_small, points_other])
        combined_variance = np.var(combined_points, axis=0).mean()
        
        # Compute variance increase
        variance_increase = (combined_variance * len(combined_points) - 
                           (cluster_stats[small_cluster_idx].variance * len(points_small) +
                            cluster_stats[other_label].variance * len(points_other)))
        
        if variance_increase < min_variance_increase:
            min_variance_increase = variance_increase
            merge_target = other_label
    
    if merge_target is None:
        logger.warning(f"No valid merge target found for cluster {small_cluster_idx}")
    
    return small_cluster_idx, merge_target

def try_multi_split(cluster_points, connectivity_subset, min_cluster_size, max_clusters=5):
    """Try splitting cluster into 2-5 parts and return the best valid split"""
    if len(cluster_points) < 2 * min_cluster_size:
        return None, 0, 0
    
    best_split = None
    best_valid_count = 0
    best_n_clusters = 0
    
    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        if len(cluster_points) < n_clusters * min_cluster_size:
            continue  # Skip if we can't have n_clusters valid clusters
            
        try:
            split_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                connectivity=connectivity_subset
            )
            split_labels = split_clustering.fit_predict(cluster_points)
            
            # Count valid clusters (size >= min_cluster_size)
            cluster_counts = Counter(split_labels)
            valid_split_clusters = [c for c, count in cluster_counts.items() 
                                   if count >= min_cluster_size]
            valid_count = len(valid_split_clusters)
            
            if valid_count > best_valid_count:
                best_split = split_labels
                best_valid_count = valid_count
                best_n_clusters = n_clusters
                logger.debug(f"New best split: {n_clusters} clusters, {valid_count} valid")
                
        except Exception as e:
            logger.warning(f"Failed to split into {n_clusters} clusters: {e}")
            continue
    
    return (best_split, best_valid_count, best_n_clusters) if best_split is not None else (None, 0, 0)

class ConstrainedAgglomerativeClustering:
    """Agglomerative clustering with minimum cluster size constraints and connectivity-based merging"""
    def __init__(self, n_clusters=None, min_cluster_size=2, connectivity=None, n_jobs=-1):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.connectivity = connectivity
        self.labels_ = None
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
    def _merge_small_clusters(self, X, labels):
        """Merge clusters smaller than min_cluster_size into their nearest neighbors"""
        iteration = 0
        
        while True:
            # Count cluster sizes
            cluster_sizes = Counter(labels)
            small_clusters = {c for c, size in cluster_sizes.items() 
                            if size < self.min_cluster_size}

            logger.info(f"Iteration {iteration}: {len(small_clusters)} small clusters remaining")
            
            if not small_clusters:
                break
            
            # Find merge targets for all small clusters in parallel
            merger_finder = partial(find_merge_target_connectivity, 
                                    X=X, 
                                    labels=labels, 
                                    small_clusters=small_clusters, 
                                    connectivity=self.connectivity)
            
            try:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(merger_finder)(small_cluster) for small_cluster in small_clusters
                )
            except Exception as e:
                logger.error(f"Parallelization failed: {e}. Falling back to sequential processing.")
                results = [merger_finder(small_cluster) for small_cluster in small_clusters]
            
            # Apply merges
            successful_merges = 0
            for small_cluster, merge_target in results:
                if merge_target is not None:
                    mask_small = labels == small_cluster
                    if np.any(mask_small):
                        labels[mask_small] = merge_target
                        successful_merges += 1
            
            logger.info(f"Successfully merged {successful_merges} clusters")
            
            # Break if no successful merges
            if successful_merges == 0:
                logger.warning("No more clusters could be merged")
                break
            
            iteration += 1
            
            # Check for infinite loops
            if iteration > len(small_clusters):
                logger.warning(f"Unable to merge all small clusters after {iteration} iterations")
                break
                
        return labels
    
    def _split_and_merge_cluster_by_id(self, X, labels, current_max_label, cluster_id):
        """Split a specific cluster and handle resulting small subclusters"""
        mask_target = labels == cluster_id
        points_target = X[mask_target]
        indices_target = np.where(mask_target)[0]
        
        if len(points_target) == 0:
            logger.warning(f"Cluster {cluster_id} has no points")
            return labels, current_max_label, 0
        
        # Get connectivity subset if available
        connectivity_subset = None
        if self.connectivity is not None:
            try:
                connectivity_subset = self.connectivity[mask_target][:, mask_target]
            except Exception as e:
                logger.warning(f"Error creating connectivity subset: {e}")
        
        # Try multiple split configurations
        max_possible_clusters = min(5, len(points_target) // self.min_cluster_size)
        split_labels, valid_count, best_n = try_multi_split(
            points_target, 
            connectivity_subset, 
            self.min_cluster_size, 
            max_clusters=max_possible_clusters
        )
        
        if split_labels is None or valid_count <= 1:
            logger.info(f"No valid split found for cluster {cluster_id}")
            return labels, current_max_label, 0
        
        logger.info(f"Split cluster {cluster_id} into {best_n} parts with {valid_count} valid subclusters")
        
        # Copy labels to avoid modifying the original
        new_labels = np.array(labels, copy=True)
        
        # Count size of each subcluster
        subcluster_counts = Counter(split_labels)
        valid_subclusters = [sc for sc, count in subcluster_counts.items() 
                           if count >= self.min_cluster_size]
        small_subclusters = [sc for sc, count in subcluster_counts.items() 
                           if count < self.min_cluster_size]
        
        if not valid_subclusters:
            return labels, current_max_label, 0
        
        # First subcluster keeps original label
        first_valid = valid_subclusters[0]
        
        # Assign new labels to remaining valid subclusters
        next_label = current_max_label + 1
        for subcluster_id in valid_subclusters[1:]:
            subcluster_mask = split_labels == subcluster_id
            if np.any(subcluster_mask):
                subcluster_indices = indices_target[subcluster_mask]
                new_labels[subcluster_indices] = next_label
                next_label += 1
        
        # Handle small subclusters
        small_subcluster_map = {}
        for subcluster_id in small_subclusters:
            subcluster_mask = split_labels == subcluster_id
            if np.any(subcluster_mask):
                subcluster_indices = indices_target[subcluster_mask]
                subcluster_points = X[subcluster_indices]
                
                small_subcluster_map[next_label] = {
                    'indices': subcluster_indices,
                    'points': subcluster_points
                }
                new_labels[subcluster_indices] = next_label
                next_label += 1
        
        # Merge small subclusters
        if small_subcluster_map:
            temp_labels = new_labels.copy()
            small_clusters_to_merge = set(small_subcluster_map.keys())
            
            for small_label, data in small_subcluster_map.items():
                if small_label in small_clusters_to_merge:
                    _, merge_target = find_merge_target_connectivity(
                        small_cluster_idx=small_label,
                        X=X,
                        labels=temp_labels,
                        small_clusters=small_clusters_to_merge,
                        connectivity=self.connectivity
                    )
                    
                    if merge_target is not None:
                        new_labels[data['indices']] = merge_target
                        temp_labels[data['indices']] = merge_target
                        small_clusters_to_merge.remove(small_label)
        
        current_max_label = next_label - 1
        valid_clusters = len(valid_subclusters) - 1  # -1 because first keeps original label
        
        return new_labels, current_max_label, valid_clusters
    
    def _adjust_n_clusters(self, X, labels):
        """Adjust cluster count by splitting large clusters"""
        if self.n_clusters is None:
            return labels
        
        current_n_clusters = len(set(labels))
        if current_n_clusters >= self.n_clusters:
            return labels
            
        self.labels_ = labels.copy()
        current_max_label = max(labels) if len(labels) > 0 else -1
        
        logger.info(f"Attempting to reach {self.n_clusters} clusters from current {current_n_clusters}")
        
        attempts = 0
        max_attempts = min(self.n_clusters - current_n_clusters + 5, 20)
        
        while current_n_clusters < self.n_clusters and attempts < max_attempts:
            prev_labels = labels.copy()
            
            # Get clusters by size
            cluster_sizes = Counter(labels)
            if not cluster_sizes:
                break
                
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            added_any_clusters = False
            
            # Try splitting top 3 largest clusters
            for cluster_id, size in sorted_clusters[:3]:
                if size < 2 * self.min_cluster_size:
                    continue
                
                temp_labels = labels.copy()
                temp_labels, new_max_label, added_clusters = self._split_and_merge_cluster_by_id(
                    X, temp_labels, current_max_label, cluster_id
                )
                
                if added_clusters > 0:
                    labels = temp_labels
                    current_max_label = new_max_label
                    added_any_clusters = True
                    break
            
            if not added_any_clusters or np.array_equal(prev_labels, labels):
                break
                
            current_n_clusters = len(set(labels))
            logger.info(f"Current cluster count: {current_n_clusters}")
            attempts += 1
            
            yield labels
    
    def fit_predict(self, X):
        """Perform constrained clustering and return labels"""
        # Handle edge cases
        if len(X) == 0:
            self.labels_ = np.array([])
            return np.array([])
        
        if len(X) == 1:
            self.labels_ = np.array([0])
            return np.array([0])
        
        # Initial clustering
        initial_n_clusters = self.n_clusters or max(min(len(X) // self.min_cluster_size, 100), 1)
        initial_n_clusters = min(initial_n_clusters, len(X) - 1)

        logger.info(f"Starting with initial_n_clusters: {initial_n_clusters}")
        
        # Handle connectivity matrix
        if self.connectivity is not None:
            try:
                n_components, component_labels = connected_components(
                    csr_matrix(self.connectivity),
                    directed=False
                )
                
                component_sizes = Counter(component_labels)
                if min(component_sizes.values()) < self.min_cluster_size:
                    logger.warning("Some connected components are smaller than min_cluster_size")
                    self.connectivity = None
            except Exception as e:
                logger.warning(f"Error processing connectivity matrix: {e}")
                self.connectivity = None
        
        # Perform initial clustering
        try:
            clustering = AgglomerativeClustering(
                n_clusters=initial_n_clusters,
                connectivity=self.connectivity
            )
            
            labels = clustering.fit_predict(X)
        except Exception as e:
            logger.error(f"Initial clustering failed: {e}")
            fallback_n_clusters = min(10, len(X) // 2)
            clustering = AgglomerativeClustering(n_clusters=fallback_n_clusters)
            labels = clustering.fit_predict(X)
        
        # Merge small clusters
        labels = self._merge_small_clusters(X, labels)
        
        # Adjust number of clusters if needed
        labels = self._adjust_n_clusters(X, labels)
        
        # Relabel clusters to be consecutive integers
        #unique_labels = np.unique(labels)
        #label_map = {old: new for new, old in enumerate(unique_labels)}
        #labels = np.array([label_map[l] for l in labels])
        
        self.labels_ = labels
        
        return labels


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    X = np.random.rand(500, 2)
    
    # Create connectivity matrix
    try:
        from sklearn.neighbors import kneighbors_graph
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False).toarray()
    except Exception as e:
        logger.warning(f"Could not create connectivity graph: {e}")
        connectivity = None
    
    # Perform constrained clustering
    clustering = ConstrainedAgglomerativeClustering(
        n_clusters=10,
        min_cluster_size=10,
        connectivity=connectivity,
        n_jobs=4
    )
    
    labels = clustering.fit_predict(X)
    
    # Print results
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"Final clusters: {len(unique_labels)}")
    for label, count in zip(unique_labels, counts):
        logger.info(f"Cluster {label}: {count} points")
