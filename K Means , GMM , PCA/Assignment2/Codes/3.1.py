import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
    
    def _compute_distances(self, X, centroids):
        """
        Compute the Euclidean distance between each point and each centroid.
        X: numpy array of shape (n_samples, n_features)
        centroids: numpy array of shape (k, n_features)
        """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances
    
    def fit(self, X):
        """
        Fit the K-Means model.
        X: numpy array of shape (n_samples, n_features)
        """
        self.X = X  # Store X as an instance variable
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from the dataset
        initial_indices = np.random.choice(n_samples, self.k, replace=False)
        initial_centroids = X[initial_indices]  # Save initial centroids for debugging
        self.centroids = X[initial_indices]
        
        for i in range(self.max_iter):
            # Assign clusters
            distances = self._compute_distances(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Compute new centroids
            new_centroids = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.k)])
            
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            
            self.centroids = new_centroids
        
       # # Print initial and final centroids for debugging
        # print("Initial Centroids:\n", initial_centroids)
        # print("Final Centroids:\n", self.centroids)
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        X: numpy array of shape (n_samples, n_features)
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def getCost(self):
        """
        Compute the Within Cluster Sum of Squares (WCSS) cost.
        """
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Model has not been fit yet.")
        
        cost = 0.0
        for i in range(self.k):
            cluster_points = self.X[self.labels_ == i]
            cost += np.sum((cluster_points - self.centroids[i]) ** 2)
        return cost

# Load data
df = pd.read_feather('word-embeddings.feather')
X = np.array(df.iloc[:, 1].tolist())  # Assuming embeddings are stored as lists
X = np.vstack(X)  # Convert lists to a 2D numpy array if necessary

# Instantiate and fit the KMeans model
kmeans = KMeans(k=6)  
kmeans.fit(X)

# Predict cluster assignments
labels = kmeans.predict(X)

# Get cost
cost = kmeans.getCost()
print(f"WCSS: {cost}")



# observation: if k is increasing the wcss is decreasing
# getting wccs different every time because the clusters are randomly selected