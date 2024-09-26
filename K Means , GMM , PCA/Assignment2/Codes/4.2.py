import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, k, max_iter=100, tol=1e-6):
        self.k = k  # Number of clusters
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        # Initialize means randomly from data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[random_indices]
        
        # Initialize covariance matrices as small identity matrices for each component
        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        
        # Initialize weights uniformly
        self.weights = np.ones(self.k) / self.k

    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        cov += 1e-6 * np.eye(n_features)  # Add small value to diagonal for numerical stability
        cov_inv = np.linalg.inv(cov)
        diff = X - mean
        exponent = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        norm_factor = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(cov))
        return exponent / (norm_factor + 1e-10)  # Add small value to prevent division by zero

    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.k))
        
        for i in range(self.k):
            responsibilities[:, i] = self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
        
        # Normalize responsibilities to ensure they sum to 1 across clusters
        responsibilities_sum = np.sum(responsibilities, axis=1, keepdims=True)
        self.responsibilities = responsibilities / (responsibilities_sum + 1e-10)  # Add small value to avoid division by zero
    
    def _m_step(self, X):
        n_samples, n_features = X.shape
        resp_sum = np.sum(self.responsibilities, axis=0)
        
        # Update means
        self.means = np.dot(self.responsibilities.T, X) / (resp_sum[:, None] + 1e-10)
        
        # Update covariances and weights
        for i in range(self.k):
            diff = X - self.means[i]
            weighted_diff = self.responsibilities[:, i][:, None] * diff
            self.covariances[i] = np.dot(weighted_diff.T, diff) / (resp_sum[i] + 1e-10) + 1e-6 * np.eye(n_features)  # Add small value for stability
            self.weights[i] = resp_sum[i] / n_samples
    
    def fit(self, X):
        self._initialize_parameters(X)
        
        for iteration in range(self.max_iter):
            prev_means = self.means.copy()
            
            self._e_step(X)
            self._m_step(X)
            
            # Check for convergence based on means
            if np.all(np.linalg.norm(self.means - prev_means, axis=1) < self.tol):
                print(f"Converged after {iteration+1} iterations.")
                break
    
    def getParams(self):
        return self.means, self.covariances, self.weights
    
    def getMembership(self):
        return self.responsibilities
    
    def getLikelihood(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)
        
        for i in range(self.k):
            likelihood += self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
        
        return np.sum(np.log(likelihood + 1e-10))  # Adding a small value to avoid log(0)

# Load the dataset
data = pd.read_feather('word-embeddings.feather')
print(data.head())

# Convert the 'vit' column into NumPy array of embeddings
embeddings = np.array(data['vit'].tolist(), dtype=np.float64)
X = embeddings  # Assuming X contains the word embeddings

print(f'Data shape: {X.shape}')

# Define a range for k
k_values = range(1, 11)

# Initialize lists to store BIC and AIC
bic_scores_custom = []
aic_scores_custom = []
bic_scores_sklearn = []
aic_scores_sklearn = []

for k in k_values:
    print(f"Testing Custom GMM with k={k}...")
    try:
        # Fit custom GMM
        gmm_custom = GMM(k=k)
        gmm_custom.fit(X)
        log_likelihood = gmm_custom.getLikelihood(X)
        num_params = k * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2 + 1)  # Means + Covariances + Weights
        bic = -2 * log_likelihood + num_params * np.log(X.shape[0])
        aic = -2 * log_likelihood + 2 * num_params
        bic_scores_custom.append(bic)
        aic_scores_custom.append(aic)
    except Exception as e:
        # print(f"Custom GMM failed for k={k}: {e}")
        bic_scores_custom.append(np.inf)
        aic_scores_custom.append(np.inf)

    # print(f"Testing Sklearn GMM with k={k}...")
    # Fit sklearn GMM for comparison
    gmm_sklearn = GaussianMixture(n_components=k, random_state=0)
    gmm_sklearn.fit(X)
    bic_scores_sklearn.append(gmm_sklearn.bic(X))
    aic_scores_sklearn.append(gmm_sklearn.aic(X))

# Plot BIC and AIC for both custom GMM and sklearn GMM
plt.figure(figsize=(12, 6))
plt.plot(k_values, bic_scores_sklearn, marker='o', linestyle='--', label='BIC (custom)')
plt.plot(k_values, aic_scores_sklearn, marker='o', linestyle='--', label='AIC (custom)')
plt.ylabel('Score')
plt.title('BIC and AIC Scores for Different k  (custom GMM)')
plt.legend()
plt.show()

# Plot BIC and AIC for both custom GMM and sklearn GMM
plt.figure(figsize=(12, 6))
plt.plot(k_values, bic_scores_sklearn, marker='o', linestyle='--', label='BIC (Sklearn)')
plt.plot(k_values, aic_scores_sklearn, marker='o', linestyle='--', label='AIC (Sklearn)')
# plt.plot(k_values, bic_scores_custom, marker='x', linestyle='-', label='BIC (Custom GMM)')
# plt.plot(k_values, aic_scores_custom, marker='x', linestyle='-', label='AIC (Custom GMM)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Score')
plt.title('BIC and AIC Scores for Different k  (Sklearn GMM)')
plt.legend()
plt.show()

# Find optimal number of clusters based on BIC and AIC
optimal_k_bic_sklearn = k_values[np.argmin(bic_scores_sklearn)]
optimal_k_aic_sklearn = k_values[np.argmin(aic_scores_sklearn)]

print(f'Optimal number of clusters  GMM BIC: {optimal_k_bic_sklearn}')
print(f'Optimal number of clusters  GMM AIC: {optimal_k_aic_sklearn}')
print (f'kgmm1:{optimal_k_bic_sklearn}')