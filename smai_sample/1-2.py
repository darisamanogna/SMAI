import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load dataset
df_tracks = pd.read_csv('spotify.csv')
df_tracks = df_tracks.drop_duplicates(subset=['track_id'], keep='first')
df_tracks.fillna(df_tracks.mean(numeric_only=True), inplace=True)  # Fill missing values

# Define feature columns and target
X = df_tracks[['danceability', 'energy', 'loudness', 'tempo', 'duration_ms']].values
y = df_tracks['track_genre'].values

# Encode target labels
def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in labels])
    return encoded_labels, label_to_index

def decode_labels(encoded_labels, index_to_label):
    return np.array([index_to_label[index] for index in encoded_labels])

y_encoded, label_to_index = encode_labels(y)

# Split data (80:10:10)
def train_val_test_split(X, y, train_size=0.8, val_size=0.1):
    assert train_size + val_size < 1.0, "Invalid split ratio."
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_end = int(train_size * len(X))
    val_end = int((train_size + val_size) * len(X))
    
    X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
    X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
    X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y_encoded)

# Implement distance functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define KNN class
class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_function = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance
        }[distance_metric]
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = np.array([self.distance_function(test_point, x_train) for x_train in self.X_train])
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            predictions.append(np.bincount(nearest_labels).argmax())
        return np.array(predictions)
    
    def set_params(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_function = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance
        }[distance_metric]

# Define function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Task 1: Find the best {k, distance metric} pair
k_values = [1, 3, 5, 7, 9]
distance_metrics = ['euclidean', 'manhattan', 'cosine']
results = []

for k in k_values:
    for metric in distance_metrics:
        knn = KNN(k=k, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_val_pred = knn.predict(X_val)
        accuracy = calculate_accuracy(y_val, y_val_pred)
        results.append((k, metric, accuracy))

# Task 2: Rank the top 10 pairs
results.sort(key=lambda x: x[2], reverse=True)
top_10_pairs = results[:10]

print("Top 10 {k, distance metric} pairs by validation accuracy:")
for k, metric, accuracy in top_10_pairs:
    print(f"k={k}, metric={metric}, accuracy={accuracy:.4f}")

# Task 3: Plot k vs accuracy for a fixed distance metric
fixed_metric = 'euclidean'
accuracies = [result[2] for result in results if result[1] == fixed_metric]

plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o')
plt.title(f'Accuracy vs. k for {fixed_metric} distance')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Task 4: Experiment with dropping various columns
columns = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms']
best_subset_accuracy = 0
best_subset = None

for subset in itertools.combinations(columns, len(columns) - 1):
    X_subset = df_tracks[list(subset)].values
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_subset, y_encoded)
    
    knn = KNN(k=top_10_pairs[0][0], distance_metric=top_10_pairs[0][1])
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    accuracy = calculate_accuracy(y_val, y_val_pred)
    
    if accuracy > best_subset_accuracy:
        best_subset_accuracy = accuracy
        best_subset = subset

print(f"Best subset: {best_subset}, Accuracy: {best_subset_accuracy:.4f}")

# Task 5: [Bonus] Try all combinations of columns
best_combination_accuracy = 0
best_combination = None

for r in range(1, len(columns) + 1):
    for subset in itertools.combinations(columns, r):
        X_subset = df_tracks[list(subset)].values
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_subset, y_encoded)
        
        knn = KNN(k=top_10_pairs[0][0], distance_metric=top_10_pairs[0][1])
        knn.fit(X_train, y_train)
        y_val_pred = knn.predict(X_val)
        accuracy = calculate_accuracy(y_val, y_val_pred)
        
        if accuracy > best_combination_accuracy:
            best_combination_accuracy = accuracy
            best_combination = subset

print(f"Best feature combination: {best_combination}, Accuracy: {best_combination_accuracy:.4f}")
