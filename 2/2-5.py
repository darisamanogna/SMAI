import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def set_params(self, k=None, distance_metric=None):
        if k is not None:
            self.k = k
        if distance_metric is not None:
            self.distance_metric = distance_metric

    def get_params(self):
        return {'k': self.k, 'distance_metric': self.distance_metric}

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, batch_size=1000):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        predictions = np.zeros(num_test, dtype=int)

        for start in range(0, num_test, batch_size):
            end = min(start + batch_size, num_test)
            batch = X[start:end]

            if self.distance_metric == 'euclidean':
                distances = np.linalg.norm(self.X_train[:, np.newaxis] - batch, axis=2)
            elif self.distance_metric == 'manhattan':
                distances = np.abs(self.X_train[:, np.newaxis] - batch).sum(axis=2)
            elif self.distance_metric == 'cosine':
                norms_X_train = np.linalg.norm(self.X_train, axis=1)
                norms_X = np.linalg.norm(batch, axis=1)
                dot_products = self.X_train @ batch.T
                distances = 1 - (dot_products / (norms_X_train[:, np.newaxis] * norms_X))
            else:
                raise ValueError("Unsupported distance metric")

            for i in range(start, end):
                distances_i = distances[:, i - start]
                predictions[i] = self._predict(distances_i)

        return predictions

    def _predict(self, distances):
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return self._most_common(k_nearest_labels)

    def _most_common(self, labels):
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = np.unique(y_true)

    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)

    def precision(self, average='macro'):
        precision = {}
        for c in self.classes:
            true_positive = np.sum((self.y_true == c) & (self.y_pred == c))
            false_positive = np.sum((self.y_true != c) & (self.y_pred == c))
            precision[c] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        
        if average == 'macro':
            return np.mean(list(precision.values()))
        elif average == 'micro':
            true_positive = np.sum(self.y_true == self.y_pred)
            false_positive = np.sum(self.y_pred != self.y_true)
            return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    def recall(self, average='macro'):
        recall = {}
        for c in self.classes:
            true_positive = np.sum((self.y_true == c) & (self.y_pred == c))
            false_negative = np.sum((self.y_true == c) & (self.y_pred != c))
            recall[c] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        if average == 'macro':
            return np.mean(list(recall.values()))
        elif average == 'micro':
            true_positive = np.sum(self.y_true == self.y_pred)
            false_negative = np.sum(self.y_true != self.y_pred)
            return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    def f1_score(self, average='macro'):
        precision = self.precision(average=average)
        recall = self.recall(average=average)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def measure_inference_time(model, X_test):
    start_time = time.time()
    model.predict(X_test)
    return time.time() - start_time

def train_test_split(X, y, test_size=0.2):
    num_test = int(len(X) * test_size)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in labels])
    return encoded_labels, label_to_index

def decode_labels(encoded_labels, label_to_index):
    index_to_label = {index: label for label, index in label_to_index.items()}
    decoded_labels = np.array([index_to_label[index] for index in encoded_labels])
    return decoded_labels


# Load and preprocess dataset
df_tracks = pd.read_csv(r"C:\Users\daris\smai-m24-assignments-darisamanogna\data\external\spotify.csv")

# Remove samples of different genres for the same song, keeping only the first sample
df_tracks = df_tracks.drop_duplicates(subset=['track_id'], keep='first')

# Check for null values
print(pd.isnull(df_tracks).sum())

# Encode the categorical labels manually
df_tracks['track_genre_encoded'], label_to_index = encode_labels(df_tracks['track_genre'])

# Define numerical features and target variable
numerical_features = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms', 'popularity']
X = df_tracks[numerical_features].values

# Manually scale the features
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_scaled = (X - mean) / std

# Define target variable
y = df_tracks['track_genre_encoded'].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

# Initialize models
knn_initial = KNN(k=20, distance_metric='euclidean')
knn_initial.fit(X_train, y_train)

# Example configurations for other models
best_knn_model = KNN(k=10, distance_metric='manhattan')  # Example parameter
most_optimized_knn_model = KNN(k=5, distance_metric='cosine')  # Example parameter

best_knn_model.fit(X_train, y_train)
most_optimized_knn_model.fit(X_train, y_train)

# Measure inference times
time_initial = measure_inference_time(knn_initial, X_val)
time_best = measure_inference_time(best_knn_model, X_val)
time_optimized = measure_inference_time(most_optimized_knn_model, X_val)

# Plot inference time
plt.figure(figsize=(10, 6))
plt.bar(['Initial KNN', 'Best KNN', 'Optimized KNN'], 
        [time_initial, time_best, time_optimized], 
        color=['blue', 'green', 'red'])
plt.xlabel('KNN Model')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time of Different KNN Models')
plt.show()

# Plot inference time vs train dataset size
train_sizes = [100, 200, 400, 800, 1600]  # Example sizes
times_initial = []
times_best = []
times_optimized = []

for size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_scaled[:size], y[:size], test_size=0.2)
    
    knn_initial.fit(X_train_subset, y_train_subset)
    times_initial.append(measure_inference_time(knn_initial, X_val[:size]))
    
    best_knn_model.fit(X_train_subset, y_train_subset)
    times_best.append(measure_inference_time(best_knn_model, X_val[:size]))
    
    most_optimized_knn_model.fit(X_train_subset, y_train_subset)
    times_optimized.append(measure_inference_time(most_optimized_knn_model, X_val[:size]))

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, times_initial, label='Initial KNN', marker='o')
plt.plot(train_sizes, times_best, label='Best KNN', marker='o')
plt.plot(train_sizes, times_optimized, label='Optimized KNN', marker='o')
plt.xlabel('Train Dataset Size')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Train Dataset Size')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate and compare models
y_pred_initial = knn_initial.predict(X_val)
y_pred_best = best_knn_model.predict(X_val)
y_pred_optimized = most_optimized_knn_model.predict(X_val)

metrics_initial = Metrics(y_val, y_pred_initial)
metrics_best = Metrics(y_val, y_pred_best)
metrics_optimized = Metrics(y_val, y_pred_optimized)

print("Initial KNN Metrics:")
print(f"Accuracy: {metrics_initial.accuracy()}")
print(f"Precision: {metrics_initial.precision()}")
print(f"Recall: {metrics_initial.recall()}")
print(f"F1 Score: {metrics_initial.f1_score()}")

print("Best KNN Metrics:")
print(f"Accuracy: {metrics_best.accuracy()}")
print(f"Precision: {metrics_best.precision()}")
print(f"Recall: {metrics_best.recall()}")
print(f"F1 Score: {metrics_best.f1_score()}")

print("Optimized KNN Metrics:")
print(f"Accuracy: {metrics_optimized.accuracy()}")
print(f"Precision: {metrics_optimized.precision()}")
print(f"Recall: {metrics_optimized.recall()}")
print(f"F1 Score: {metrics_optimized.f1_score()}")
