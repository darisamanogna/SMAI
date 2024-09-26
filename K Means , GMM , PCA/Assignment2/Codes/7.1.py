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


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_feather('word-embeddings.feather')

# Convert the 'vit' column into NumPy array of embeddings
embeddings = np.array(data['vit'].tolist(), dtype=np.float64)
X = embeddings  # X contains the word embeddings

# Define k values for clustering
k_values = [2, 3, 5]  # Adjust as needed

# Function to perform K-Means clustering and evaluate coherence
def perform_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    return labels, silhouette_avg

# Function to perform GMM clustering and evaluate coherence
def perform_gmm(X, k):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    return labels, silhouette_avg

# Perform K-Means clustering
kmeans_results = {}
for k in k_values:
    labels, silhouette_avg = perform_kmeans(X, k)
    kmeans_results[k] = {
        'labels': labels,
        'silhouette_score': silhouette_avg
    }
    print(f'K-Means for k={k}: Silhouette Score = {silhouette_avg}')

# Perform GMM clustering
gmm_results = {}
for k in k_values:
    labels, silhouette_avg = perform_gmm(X, k)
    gmm_results[k] = {
        'labels': labels,
        'silhouette_score': silhouette_avg
    }
    print(f'GMM for k={k}: Silhouette Score = {silhouette_avg}')

# Plot comparison of K-Means and GMM silhouette scores
def plot_comparison(kmeans_results, gmm_results):
    plt.figure(figsize=(12, 6))
    kmeans_scores = [result['silhouette_score'] for result in kmeans_results.values()]
    gmm_scores = [result['silhouette_score'] for result in gmm_results.values()]
    k_values = list(kmeans_results.keys())
    
    plt.plot(k_values, kmeans_scores, marker='o', linestyle='-', label='K-Means')
    plt.plot(k_values, gmm_scores, marker='o', linestyle='-', label='GMM')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Comparison of K-Means and GMM')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_comparison(kmeans_results, gmm_results)

# Determine the best k for K-Means and GMM based on silhouette scores
best_kmeans_k = max(kmeans_results, key=lambda k: kmeans_results[k]['silhouette_score'])
best_gmm_k = max(gmm_results, key=lambda k: gmm_results[k]['silhouette_score'])

print(f'Best k for K-Means: {best_kmeans_k}')
print(f'Best k for GMM: {best_gmm_k}')

# Function to examine clusters and their contents
def examine_clusters(X, labels):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        print(f'Cluster {label}: {cluster_points.shape[0]} points')

# Function to visualize clusters for each k value
def plot_clusters_all(k_results, method_name):
    for k, result in k_results.items():
        labels = result['labels']
        plt.figure(figsize=(12, 6))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter(X[labels == label, 0], X[labels == label, 1], label=f'Cluster {label}')
        plt.title(f'{method_name} Clustering for k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

# Assuming X has at least 2 features for visualization
print("Visualizing K-Means clusters for all k values:")
plot_clusters_all(kmeans_results, 'K-Means')

print("Visualizing GMM clusters for all k values:")
plot_clusters_all(gmm_results, 'GMM')
