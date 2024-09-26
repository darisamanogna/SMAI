import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def _compute_distances(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances
    
    def fit(self, X):
        self.X = X  # Store X as an instance variable
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from the dataset
        initial_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[initial_indices]
        
        for i in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Compute new centroids
            new_centroids = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.k)])
            
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            
            self.centroids = new_centroids
        
    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def getCost(self):
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Model has not been fit yet.")
        
        cost = 0.0
        for i in range(self.k):
            cluster_points = self.X[self.labels_ == i]
            cost += np.sum((cluster_points - self.centroids[i]) ** 2)
        return cost

class ElbowMethod:
    def __init__(self, X, k_range, random_state=None):
        self.X = X
        self.k_range = k_range
        self.wcss = []
        self.random_state = random_state
    
    def compute_wcss(self):
        for k in self.k_range:
            kmeans = KMeans(k=k, max_iter=300, tol=1e-4, random_state=self.random_state)
            kmeans.fit(self.X)
            cost = kmeans.getCost()
            self.wcss.append(cost)
            print(f'Number of clusters: {k}, WCSS: {cost}')
    
    def plot_elbow(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.k_range, self.wcss, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal k')
        plt.xticks(self.k_range)
        plt.grid(True)
        plt.show()
    
    def find_elbow(self):
        """
        Automatically find the elbow point by calculating the maximum
        distance between the line formed by (first_k, first_wcss) and 
        (last_k, last_wcss), and each point in the WCSS plot.
        """
        p1 = np.array([self.k_range[0], self.wcss[0]])
        p2 = np.array([self.k_range[-1], self.wcss[-1]])
        
        distances = []
        for i in range(len(self.k_range)):
            p = np.array([self.k_range[i], self.wcss[i]])
            distance = np.abs(np.cross(p2-p1, p-p1)) / np.linalg.norm(p2-p1)
            distances.append(distance)
        
        # Return the index of the maximum distance (elbow point)
        optimal_k_index = np.argmax(distances)
        return self.k_range[optimal_k_index]

# Load data
df = pd.read_feather('word-embeddings.feather')

# Extract embeddings
embeddings = df.iloc[:, 1].tolist()
if isinstance(embeddings[0], np.ndarray):
    X = np.array(embeddings)
else:
    def parse_embedding(embedding):
        return np.fromstring(embedding.strip('[]'), sep=',')
    X = np.array([parse_embedding(embedding) for embedding in embeddings])

# Define a range for k
k_values = range(1, 10)  # Adjust as needed

# Use ElbowMethod to compute and plot WCSS
elbow_method = ElbowMethod(X, k_values, random_state=42)
elbow_method.compute_wcss()
elbow_method.plot_elbow()

# Choose the optimal number of clusters (kkmeans1) based on the elbow plot
# kkmeans1 = int(input("Enter the optimal number of clusters (kkmeans1) based on the elbow plot: "))

# Perform K-means clustering using the optimal number of clusters
kmeans = KMeans(k=3, max_iter=300, tol=1e-4, random_state=42)
kmeans.fit(X)

# Predict cluster assignments
clusters = kmeans.predict(X)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Print cluster values for the data samples
print("Cluster labels for data samples:")
print(df[['cluster']].head(200))
