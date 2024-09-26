import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time

# Load the data
data = pd.read_feather('word-embeddings.feather')

# Convert the 'vit' column into NumPy array of embeddings
embeddings = np.array(data['vit'].tolist(), dtype=np.float64)
X = embeddings  # X contains the word embeddings

# For evaluation, we need labels. Assuming you have labels; let's create a dummy example:
# Replace with actual labels if you have them in your dataset.
# Dummy labels for demonstration (e.g., creating clusters or class labels manually)
y = np.random.randint(0, 5, X.shape[0])  # Randomly assign 5 classes

# Split the dataset into training and validation subsets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 9.1 PCA + KNN
def perform_pca(X_train, X_val, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_reduced = pca.fit_transform(X_train)
    X_val_reduced = pca.transform(X_val)
    return X_train_reduced, X_val_reduced

def knn_classification(X_train, y_train, X_val, y_val, k, metric):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    start_time = time.time()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_val)
    inference_time = time.time() - start_time
    return predictions, inference_time

# Function to calculate and print metrics
def evaluate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return f1, accuracy, precision, recall

# Generate scree plot to determine optimal number of dimensions
def scree_plot(X):
    pca = PCA().fit(X)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

# Determine optimal number of dimensions using scree plot
scree_plot(X)

# Example of choosing an optimal number of components (e.g., 10) based on the scree plot
optimal_n_components = 10  # Replace with the optimal number of dimensions based on the scree plot

# Perform PCA
X_train_reduced, X_val_reduced = perform_pca(X_train, X_val, optimal_n_components)

# Perform KNN on original dataset
k = 5  # Example k value
metric = 'euclidean'  # Example distance metric

# KNN on original dataset
original_start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
knn.fit(X_train, y_train)
y_pred_original = knn.predict(X_val)
original_inference_time = time.time() - original_start_time

# KNN on reduced dataset
y_pred_reduced, reduced_inference_time = knn_classification(X_train_reduced, y_train, X_val_reduced, y_val, k, metric)

# Evaluate metrics
print("Metrics for KNN on original dataset:")
f1_original, accuracy_original, precision_original, recall_original = evaluate_metrics(y_val, y_pred_original)
print(f"F1 Score: {f1_original}")
print(f"Accuracy: {accuracy_original}")
print(f"Precision: {precision_original}")
print(f"Recall: {recall_original}")

print("\nMetrics for KNN on reduced dataset:")
f1_reduced, accuracy_reduced, precision_reduced, recall_reduced = evaluate_metrics(y_val, y_pred_reduced)
print(f"F1 Score: {f1_reduced}")
print(f"Accuracy: {accuracy_reduced}")
print(f"Precision: {precision_reduced}")
print(f"Recall: {recall_reduced}")

# Plot inference times
plt.figure(figsize=(10, 6))
plt.bar(['Original Dataset', 'Reduced Dataset'], [original_inference_time, reduced_inference_time], color=['blue', 'orange'])
plt.xlabel('Dataset')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time Comparison')
plt.grid(True)
plt.show()

# Compare metrics
print("\nComparison of Metrics:")
print(f"F1 Score Improvement: {f1_reduced - f1_original}")
print(f"Accuracy Improvement: {accuracy_reduced - accuracy_original}")
print(f"Precision Improvement: {precision_reduced - precision_original}")
print(f"Recall Improvement: {recall_reduced - recall_original}")
