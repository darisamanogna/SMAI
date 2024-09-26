import numpy as np
import pandas as pd

# Load data from CSV files
def load_data_from_files(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Ensure all data is numeric
    for df in [train_df, val_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'object':  # Check if the column is of type object (string)
                df[col] = pd.Categorical(df[col]).codes  # Convert to categorical codes
    
    # Extract features and labels
    X_train = train_df.drop('track_genre', axis=1).values.astype(float)
    y_train = train_df['track_genre'].values
    X_val = val_df.drop('track_genre', axis=1).values.astype(float)
    y_val = val_df['track_genre'].values
    X_test = test_df.drop('track_genre', axis=1).values.astype(float)
    y_test = test_df['track_genre'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Define distance functions
def euclidean_distance(X1, X2):
    return np.sqrt(np.sum((X1 - X2) ** 2, axis=1))

def manhattan_distance(X1, X2):
    return np.sum(np.abs(X1 - X2), axis=1)

def cosine_distance(X1, X2):
    dot_product = np.sum(X1 * X2, axis=1)
    magnitude = np.sqrt(np.sum(X1 ** 2, axis=1)) * np.sqrt(np.sum(X2 ** 2, axis=1))
    return 1 - dot_product / magnitude

# KNN Classifier Implementation
class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._compute_distances(x, self.X_train)
            k_nearest = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest]
            predictions.append(np.bincount(k_nearest_labels).argmax())
        return np.array(predictions)
    
    def _compute_distances(self, X1, X2):
        X1 = np.atleast_2d(X1).astype(float)  # Ensure X1 is 2D and of type float
        X2 = X2.astype(float)  # Ensure X2 is of type float
        if self.distance_metric == 'euclidean':
            return euclidean_distance(X1, X2)
        elif self.distance_metric == 'manhattan':
            return manhattan_distance(X1, X2)
        elif self.distance_metric == 'cosine':
            return cosine_distance(X1, X2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

# Evaluation function
def evaluate(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# File paths
train_path = r"C:\Users\daris\smai-m24-assignments-darisamanogna\data\external\spotify-2\train.csv"
val_path = r"C:\Users\daris\smai-m24-assignments-darisamanogna\data\external\spotify-2\validate.csv"
test_path = r"C:\Users\daris\smai-m24-assignments-darisamanogna\data\external\spotify-2\test.csv"

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_files(train_path, val_path, test_path)

# Apply KNN for different distance metrics
k = 5  # Replace with the best k you found
metrics = ['euclidean', 'manhattan', 'cosine']

for metric in metrics:
    knn = KNN(k=k, distance_metric=metric)
    knn.fit(X_train, y_train)
    
    # Validation accuracy
    y_val_pred = knn.predict(X_val)
    val_accuracy = evaluate(y_val, y_val_pred)
    print(f"Validation Accuracy with {metric} distance: {val_accuracy:.4f}")
    
    # Test accuracy
    y_test_pred = knn.predict(X_test)
    test_accuracy = evaluate(y_test, y_test_pred)
    print(f"Test Accuracy with {metric} distance: {test_accuracy:.4f}")
    print('-' * 40)
