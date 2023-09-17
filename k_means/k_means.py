import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, n_centroids, max_iters=100):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.n_centroids = n_centroids
        self.centroids = None 
        self.max_iters = max_iters
        

    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """        
        # Initialize cluster centers randomly or using a specific initialization method
        n_samples, n_features = X.shape
        M = X.values
                
        centroid_indices = np.random.choice(n_samples, 1, replace=False)
        self.centroids = M[centroid_indices]
        min_distance_threshold = 0.22

        print(self.centroids)
        print(M)

        for _ in range(self.n_centroids - 1):
            distances = np.zeros(n_samples)
            for c in self.centroids:
                distances += np.linalg.norm(M - c, axis=1) ** 2  # Accumulate distances

            min_distances = np.min(distances)  # Find the minimum distance
            probabilities = distances / np.sum(distances)  # Calculate probabilities

            # Choose a new centroid, ensuring it's not too close to existing centroids
            new_centroid_index = None
            while new_centroid_index is None:
                potential_index = np.random.choice(n_samples, p=probabilities)
                potential_centroid = M[potential_index]
                min_distance_to_existing = np.min(np.linalg.norm(self.centroids - potential_centroid, axis=1))
                
                # Set a threshold for the minimum distance to avoid very close centroids
                if min_distance_to_existing >= min_distance_threshold:
                    new_centroid_index = potential_index

            self.centroids = np.append(self.centroids, M[new_centroid_index]).reshape(-1, n_features)

        self.centroids = np.array(self.centroids)




    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        for _ in range(self.max_iters):
            list_of_labels = []
            # Assign each data point to the nearest cluster
            for row in X.itertuples(index=True, name='Pandas'):           
                distances = euclidean_distance((row.x0,row.x1), self.centroids)
                labels = np.argmin(distances)
                list_of_labels.append(labels)
            list_of_labels = np.array(list_of_labels)

            # Update cluster centers by computing the mean of data points in each cluster
            cluster_indecies = [np.where(list_of_labels == i)[0] for i in range(self.n_centroids)]
            cluster_centers = []

            for i, indecies in enumerate(cluster_indecies):
                if len(indecies) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X.iloc[indecies], axis=0))
            
            cluster_centers = np.array(cluster_centers)

            # Check for convergence
            if np.max(np.abs(self.centroids - cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = cluster_centers

        return list_of_labels

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    def scale_data(self, X):

        X_norm = (X-X.min())/(X.max()-X.min())

        return X_norm
    



    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  