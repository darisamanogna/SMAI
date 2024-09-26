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
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
data = pd.read_feather('word-embeddings.feather')

# Print the first few rows of the dataframe to verify
print(data.head())

# Convert the 'vit' column into NumPy array of embeddings
embeddings = np.array(data['vit'].tolist(), dtype=np.float64)
X = embeddings  # X contains the word embeddings

# Check the shape of X to confirm it's correctly loaded
print(f"Shape of X: {X.shape}")

# Now you can perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)  # Example with k=3
kmeans.fit(X)
labels = kmeans.labels_

# Plot the results (assuming X is 2D for simplicity)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering')
plt.show()

# Assuming you have determined k2 from visualization
k2 = 3  # Example value

kmeans = KMeans(n_clusters=k2, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering (k2)')
plt.show()

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)

# Plot the explained variance ratio
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()
optimal_n_components = 2  # Example value, use value from scree plot

pca = PCA(n_components=optimal_n_components)
reduced_data = pca.fit_transform(X)
from sklearn.cluster import KMeans

distortions = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(reduced_data)
    distortions.append(kmeans.inertia_)

plt.plot(K, distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose optimal k from the plot
kkmeans3 = 3  # Example value
kmeans = KMeans(n_clusters=kkmeans3, random_state=0)
kmeans.fit(reduced_data)
labels_reduced = kmeans.labels_

# Plot the results
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_reduced, cmap='viridis')
plt.title('K-means Clustering on PCA-reduced Data')
plt.show()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=k2, random_state=0)
gmm.fit(X)
labels_gmm = gmm.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis')
plt.title('GMM Clustering (k2)')
plt.show()
lowest_bic = np.infty
bic = []
n_components_range = range(1, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(reduced_data)
    bic.append(gmm.bic(reduced_data))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        kgmm3 = n_components

plt.plot(n_components_range, bic, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC')
plt.title('BIC for GMM')
plt.show()
gmm = GaussianMixture(n_components=kgmm3, random_state=0)
gmm.fit(reduced_data)
labels_gmm_reduced = gmm.predict(reduced_data)

# Plot the results
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_gmm_reduced, cmap='viridis')
plt.title('GMM Clustering on PCA-reduced Data')
plt.show()
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

# Elbow Method to find optimal k for KMeans
KKmeans3 = range(1, 11)
elbow_method = ElbowMethod(X, KKmeans3)
elbow_method.compute_wcss()
elbow_method.plot_elbow()

# PCA
pca = PCA(n_components=2)
pca.fit(X)
pca_transformed = pca.transform(X)

# Plot PCA Results
plt.figure(figsize=(10, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], s=1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of the Dataset')
plt.show()

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.eigenvalues) + 1), pca.eigenvalues, marker='o', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# GMM to find the best number of clusters
aic_values = []
bic_values = []
components_range = range(1, 11)

for n_components in components_range:
    gmm = GMM(n_components=n_components)
    gmm.fit(X)
    aic_values.append(gmm.getAIC())
    bic_values.append(gmm.getBIC())

# Plot AIC and BIC
plt.figure(figsize=(12, 6))
plt.plot(components_range, aic_values, marker='o', label='AIC')
plt.plot(components_range, bic_values, marker='o', label='BIC')
plt.xlabel('Number of components')
plt.ylabel('Criterion value')
plt.title('AIC and BIC for different numbers of components in GMM')
plt.legend()
plt.show()

# Determine optimal number of components (kgmm3) based on BIC
kgmm3 = components_range[np.argmin(bic_values)]  # You can use AIC as well if preferred

# Perform K-Means Clustering with k=2
k2 = 2
kmeans = KMeans(k=k2, max_iter=300, tol=1e-4)
kmeans.fit(X)

# Predict cluster labels
cluster_labels = kmeans.predict(X)

# Visualize K-Means Clustering Results
plt.figure(figsize=(12, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cluster_labels, s=50, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering with k=2')
plt.legend()
plt.show()

# Perform K-Means Clustering with k=KKmeans3 on PCA-reduced data
# Example where kkmeans3 should be defined
kkmeans3 = 3  # Define the number of clusters or appropriate value
# kmeans_pca = KMeans(k=kkmeans3, max_iter=300, tol=1e-4)

kmeans_pca = KMeans(k=kkmeans3, max_iter=300, tol=1e-4)
kmeans_pca.fit(pca_transformed)
cluster_labels_pca = kmeans_pca.predict(pca_transformed)

# Visualize K-Means Clustering Results on PCA-reduced data
plt.figure(figsize=(12, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cluster_labels_pca, s=50, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_pca.centroids[:, 0], kmeans_pca.centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'K-Means Clustering with k={kkmeans3}')
plt.legend()
plt.show()

# Perform GMM Clustering with n_components=k2
gmm_k2 = GMM(n_components=k2)
gmm_k2.fit(X)
cluster_labels_gmm_k2 = gmm_k2.predict(X)

# Visualize GMM Clustering Results
plt.figure(figsize=(12, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cluster_labels_gmm_k2, s=50, cmap='viridis', alpha=0.7)
plt.scatter(gmm_k2.means[:, 0], gmm_k2.means[:, 1], s=200, c='red', marker='X', label='Means')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'GMM Clustering with n_components={k2}')
plt.legend()
plt.show()

# Apply GMM with kgmm3 to PCA-reduced data
gmm_kgmm3 = GMM(n_components=kgmm3)
gmm_kgmm3.fit(pca_transformed)
cluster_labels_gmm_kgmm3 = gmm_kgmm3.predict(pca_transformed)

# Visualize GMM Clustering Results on PCA-reduced data
plt.figure(figsize=(12, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cluster_labels_gmm_kgmm3, s=50, cmap='viridis', alpha=0.7)
plt.scatter(gmm_kgmm3.means[:, 0], gmm_kgmm3.means[:, 1], s=200, c='red', marker='X', label='Means')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'GMM Clustering with n_components={kgmm3}')
plt.legend()
plt.show()