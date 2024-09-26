import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Custom PCA class
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
        
        # Mirror the second principal component along y-axis for 2D
        if self.n_components == 2:
            transformed_data[:, 1] = -transformed_data[:, 1]
        
        return transformed_data
    
    def checkPCA(self):
        return self.components.shape[1] == self.n_components

# Load the dataset
df = pd.read_feather('word-embeddings.feather')

# Extract words and embeddings
words = df.iloc[:, 0].values
embeddings = df.iloc[:, 1].values

# Convert embeddings to a NumPy array
embeddings = np.array(embeddings.tolist(), dtype=float)

# Custom PCA
pca_2d = PCA(n_components=2)
pca_2d.fit(embeddings)
transformed_data_2d = pca_2d.transform(embeddings)

pca_3d = PCA(n_components=3)
pca_3d.fit(embeddings)
transformed_data_3d = pca_3d.transform(embeddings)

# Verify Custom PCA functionality
print("2D PCA check:", pca_2d.checkPCA())
print("3D PCA check:", pca_3d.checkPCA())

# Plotting 2D Custom PCA 
plt.figure(figsize=(10, 6))
plt.scatter(transformed_data_2d[:, 0], transformed_data_2d[:, 1], alpha=0.5)
plt.title('PCA - 2D Visualization (Custom)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2 (Mirrored)')
plt.show()

# Plotting 3D Custom PCA 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_data_3d[:, 0], transformed_data_3d[:, 1], transformed_data_3d[:, 2], alpha=0.5)
ax.set_title('PCA - 3D Visualization (Custom)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

# Add labels for 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_data_3d[:, 0], transformed_data_3d[:, 1], transformed_data_3d[:, 2], alpha=0.5)

for i, word in enumerate(words):
    ax.text(transformed_data_3d[i, 0], transformed_data_3d[i, 1], transformed_data_3d[i, 2], word, fontsize=8, alpha=0.7)

ax.set_title('PCA - 3D Visualization with Word Labels (Custom)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

explained_variance_ratio = pca_2d.eigenvalues / np.sum(pca_2d.eigenvalues)
print("Explained variance ratio for 2D PCA:", explained_variance_ratio)
# explained_variance_ratio = pca_2d.eigenvalues / np.sum(pca_2d.eigenvalues)
# print("Explained variance ratio for 2D PCA:", explained_variance_ratio[:2])
print("Explained variance ratio for 3D PCA:", explained_variance_ratio[:3])

# Add labels for each point
for i, word in enumerate(words):
    ax.text(transformed_data_3d[i, 0], transformed_data_3d[i, 1], transformed_data_3d[i, 2], word, fontsize=8, alpha=0.7)

plt.show()

# Calculate explained variance ratio
explained_variance_ratio = pca_2d.eigenvalues / np.sum(pca_2d.eigenvalues)
print("Explained variance ratio for 2D PCA:", explained_variance_ratio[:2])
print("Explained variance ratio for 3D PCA:", explained_variance_ratio[:3])

# Analysis of Principal Components
print("Principal Component Analysis:")
print("In 2D PCA:")
print(f"Principal Component 1 explains {explained_variance_ratio[0]:.2%} of the variance.")
print(f"Principal Component 2 explains {explained_variance_ratio[1]:.2%} of the variance.")
print("\nIn 3D PCA:")
print(f"Principal Component 1 explains {explained_variance_ratio[0]:.2%} of the variance.")
print(f"Principal Component 2 explains {explained_variance_ratio[1]:.2%} of the variance.")
print(f"Principal Component 3 explains {explained_variance_ratio[2]:.2%} of the variance.")