import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# KMeans Class
class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
    
    def _compute_distances(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances
    
    def fit(self, X):
        self.X = X
        n_samples, n_features = X.shape
        initial_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[initial_indices]
        
        for i in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.k)])
            
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

# ElbowMethod Class
class ElbowMethod:
    def __init__(self, X, k_range):
        self.X = X
        self.k_range = k_range
        self.wcss = []
    
    def compute_wcss(self):
        for k in self.k_range:
            kmeans = KMeans(k=k, max_iter=300, tol=1e-4)
            kmeans.fit(self.X)
            self.wcss.append(kmeans.getCost())
            print(f'Number of clusters: {k}, WCSS: {kmeans.getCost()}')
    
    def plot_elbow(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.k_range, self.wcss, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal k')
        plt.xticks(self.k_range)
        plt.show()

# GMM Class
class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6, init_means=None, init_covariances=None, init_weights=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means = init_means
        self.covariances = init_covariances
        self.weights = init_weights
        self.log_likelihoods = []
    
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        if self.means is None:
            random_idx = np.random.choice(n_samples, self.n_components, replace=False)
            self.means = X[random_idx]
        
        if self.covariances is None:
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        if self.weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
    
    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        cov += 1e-6 * np.eye(n_features)
        
        cov_inv = np.linalg.inv(cov)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        log_norm_factor = -0.5 * (n_features * np.log(2 * np.pi) + np.log(np.linalg.det(cov) + 1e-10))
        log_prob = exponent + log_norm_factor
        return np.exp(log_prob)
    
    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            responsibilities[:, i] = self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        for i in range(self.n_components):
            diff = X - self.means[i]
            self.covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]
            self.covariances[i] += 1e-6 * np.eye(n_features)
        
        self.weights = Nk / n_samples
    
    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for i in range(self.n_components):
            prob = self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
            log_likelihood += np.log(prob.sum() + 1e-10)
        return log_likelihood
    
    def fit(self, X):
        self._initialize_parameters(X)
        
        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            
            if iteration > 0 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                break
    
    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        return self._e_step(X)
    
    def getAIC(self):
        n_samples, n_features = self.means.shape
        num_params = self.n_components * (n_features + 0.5 * n_features * (n_features + 1)) + (self.n_components - 1)
        log_likelihood = self.log_likelihoods[-1]
        return 2 * num_params - 2 * log_likelihood
    
    def getBIC(self):
        n_samples, n_features = self.means.shape
        num_params = self.n_components * (n_features + 0.5 * n_features * (n_features + 1)) + (self.n_components - 1)
        log_likelihood = self.log_likelihoods[-1]
        return num_params * np.log(n_samples) - 2 * log_likelihood

# PCA Class
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        if np.any(np.isnan(X_centered)) or np.any(np.isinf(X_centered)):
            raise ValueError("Data contains NaNs or Infs")
        
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.eigenvalues = eigenvalues[sorted_indices]
        
        self.components = self.eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        X = np.array(X, dtype=float)
        X_centered = X - self.mean
        transformed_data = np.dot(X_centered, self.components)
        
        if self.n_components == 2:
            transformed_data[:, 1] = -transformed_data[:, 1]
        
        return transformed_data
    
    def checkPCA(self):
        return self.components.shape[1] == self.n_components

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

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

# Step 1: Compute Linkage Matrix for Different Linkage Methods
linkage_methods = ['single', 'complete', 'average', 'ward']
linkage_matrices = {}

for method in linkage_methods:
    Z = hc.linkage(X, method=method, metric='euclidean')
    linkage_matrices[method] = Z
    plt.figure(figsize=(10, 5))
    hc.dendrogram(Z)
    plt.title(f'Dendrogram for {method.capitalize()} Linkage')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

# Step 2: Select the Best Linkage Method (based on dendrogram or user preference)
best_linkage_method = 'ward'  # For example, you may choose 'ward' as the best one
Z_best = linkage_matrices[best_linkage_method]

# Step 3: Cut the Dendrogram to Form Clusters at Different k Values
# Assume kbest1 from KMeans and kbest2 from GMM clustering are provided
kbest1 = 3  # Example, from KMeans clustering result
kbest2 = 4  # Example, from GMM clustering result

# Form clusters using fcluster at the appropriate cut levels
clusters_kbest1 = fcluster(Z_best, kbest1, criterion='maxclust')
clusters_kbest2 = fcluster(Z_best, kbest2, criterion='maxclust')

# Step 4: Compare Clusters (kbest1 and kbest2) with KMeans and GMM
print(f'Clusters formed by cutting dendrogram at kbest1={kbest1}: {clusters_kbest1}')
print(f'Clusters formed by cutting dendrogram at kbest2={kbest2}: {clusters_kbest2}')

# Plot dendrogram with highlighted clusters
plt.figure(figsize=(10, 5))
hc.dendrogram(Z_best, truncate_mode='lastp', p=max(kbest1, kbest2), show_leaf_counts=True)
plt.title(f'Dendrogram for Best Linkage ({best_linkage_method.capitalize()}) with kbest1 and kbest2')
plt.axhline(y=Z_best[-kbest1, 2], color='r', linestyle='--', label=f'Cut at kbest1={kbest1}')
plt.axhline(y=Z_best[-kbest2, 2], color='g', linestyle='--', label=f'Cut at kbest2={kbest2}')
plt.legend()
plt.show()
