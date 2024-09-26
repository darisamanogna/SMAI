# import numpy as np
# import pandas as pd

# class KNN:
#     def __init__(self, k=3, distance_metric='euclidean'):
#         self.k = k
#         self.distance_metric = distance_metric

#     def set_params(self, k=None, distance_metric=None):
#         if k is not None:
#             self.k = k
#         if distance_metric is not None:
#             self.distance_metric = distance_metric

#     def get_params(self):
#         return {'k': self.k, 'distance_metric': self.distance_metric}

#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y

#     def predict(self, X):
#         return np.array([self._predict(x) for x in X])

#     def _predict(self, x):
#         distances = self._compute_distances(x)
#         k_indices = np.argsort(distances)[:self.k]
#         k_nearest_labels = [self.y_train[i] for i in k_indices]
#         return self._most_common(k_nearest_labels)

#     def _compute_distances(self, x):
#         if self.distance_metric == 'euclidean':
#             return np.linalg.norm(self.X_train - x, axis=1)
#         elif self.distance_metric == 'manhattan':
#             return np.sum(np.abs(self.X_train - x), axis=1)
#         elif self.distance_metric == 'cosine':
#             return np.array([self._cosine_distance(x, x_train) for x_train in self.X_train])
#         else:
#             raise ValueError("Unsupported distance metric")

#     def _cosine_distance(self, x1, x2):
#         dot_product = np.dot(x1, x2)
#         norm_x1 = np.linalg.norm(x1)
#         norm_x2 = np.linalg.norm(x2)
#         return 1 - (dot_product / (norm_x1 * norm_x2)) if (norm_x1 * norm_x2) != 0 else 1

#     def _most_common(self, labels):
#         labels = np.array(labels)
#         unique, counts = np.unique(labels, return_counts=True)
#         return unique[np.argmax(counts)]

# class Metrics:
#     def __init__(self, y_true, y_pred):
#         self.y_true = y_true
#         self.y_pred = y_pred
#         self.classes = np.unique(y_true)

#     def accuracy(self):
#         return np.mean(self.y_true == self.y_pred)

#     def precision(self, average='macro'):
#         precision = {}
#         for c in self.classes:
#             true_positive = np.sum((self.y_true == c) & (self.y_pred == c))
#             false_positive = np.sum((self.y_true != c) & (self.y_pred == c))
#             precision[c] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        
#         if average == 'macro':
#             return np.mean(list(precision.values()))
#         elif average == 'micro':
#             true_positive = np.sum(self.y_true == self.y_pred)
#             false_positive = np.sum(self.y_pred != self.y_true)
#             return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

#     def recall(self, average='macro'):
#         recall = {}
#         for c in self.classes:
#             true_positive = np.sum((self.y_true == c) & (self.y_pred == c))
#             false_negative = np.sum((self.y_true == c) & (self.y_pred != c))
#             recall[c] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
#         if average == 'macro':
#             return np.mean(list(recall.values()))
#         elif average == 'micro':
#             true_positive = np.sum(self.y_true == self.y_pred)
#             false_negative = np.sum(self.y_true != self.y_pred)
#             return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

#     def f1_score(self, average='macro'):
#         precision = self.precision(average=average)
#         recall = self.recall(average=average)
#         return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# def train_test_split(X, y, test_size=0.2):
#     num_test = int(len(X) * test_size)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     test_indices = indices[:num_test]
#     train_indices = indices[num_test:]
    
#     X_train, X_test = X[train_indices], X[test_indices]
#     y_train, y_test = y[train_indices], y[test_indices]

#     return X_train, X_test, y_train, y_test

# def encode_labels(labels):
#     unique_labels = np.unique(labels)
#     label_to_index = {label: index for index, label in enumerate(unique_labels)}
#     encoded_labels = np.array([label_to_index[label] for label in labels])
#     return encoded_labels, label_to_index

# def decode_labels(encoded_labels, label_to_index):
#     index_to_label = {index: label for label, index in label_to_index.items()}
#     decoded_labels = np.array([index_to_label[index] for index in encoded_labels])
#     return decoded_labels

# # Load the dataset
# # df_tracks = pd.read_csv("C:\Users\daris\smai-m24-assignments-darisamanogna\data\external\spotify.csv")
# df_tracks = pd.read_csv(r"C:\Users\daris\smai-m24-assignments-darisamanogna\data\external\spotify.csv")

# # Remove samples of different genres for the same song, keeping only the first sample
# df_tracks = df_tracks.drop_duplicates(subset=['track_id'], keep='first')

# # Check for null values
# print(pd.isnull(df_tracks).sum())

# # Info about dataset
# df_tracks.info()

# # Encode the categorical labels manually
# df_tracks['track_genre_encoded'], label_to_index = encode_labels(df_tracks['track_genre'])

# # Define numerical features and target variable
# numerical_features = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms', 'popularity']
# X = df_tracks[numerical_features].values

# # Manually scale the features
# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# X_scaled = (X - mean) / std

# # Define target variable
# y = df_tracks['track_genre_encoded'].values

# # Train-test split
# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

# # Initialize and train KNN model
# knn = KNN(k=20, distance_metric='euclidean')
# knn.fit(X_train, y_train)

# # Predict and decode
# y_pred_encoded = knn.predict(X_val)
# y_pred = decode_labels(y_pred_encoded, label_to_index)
# y_val_decoded = decode_labels(y_val, label_to_index)

# # Metrics
# metrics = Metrics(y_val_decoded, y_pred)

# print("Accuracy:", metrics.accuracy())
# print("Precision:", metrics.precision())
# print("Recall:", metrics.recall())
# print("F1 Score:", metrics.f1_score())


import numpy as np
import pandas as pd
import pathlib

def load_and_prepare_data():
    # Define file paths
    data_path = "C:\\Users\\daris\\smai-m24-assignments-darisamanogna\\data\\external\\spotify.csv"
    interim_folder = "data/interim"
    train_path = f"{interim_folder}/spot_train.csv"
    val_path = f"{interim_folder}/spot_val.csv"
    test_path = f"{interim_folder}/spot_test.csv"

    # Create interim folder if it doesn't exist
    pathlib.Path(interim_folder).mkdir(parents=True, exist_ok=True)

    # Load the dataset
    df_tracks = pd.read_csv(data_path)

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

    # Check if split files exist
    if not (pathlib.Path(train_path).exists() and pathlib.Path(val_path).exists() and pathlib.Path(test_path).exists()):
        # Train-test split
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

        # Save splits into separate CSV files
        train_df = pd.DataFrame(X_train, columns=numerical_features)
        train_df['track_genre_encoded'] = y_train
        train_df.to_csv(train_path, index=False)

        val_df = pd.DataFrame(X_val, columns=numerical_features)
        val_df['track_genre_encoded'] = y_val
        val_df.to_csv(val_path, index=False)

        test_df = pd.DataFrame(X_test, columns=numerical_features)
        test_df['track_genre_encoded'] = y_test
        test_df.to_csv(test_path, index=False)

        print(f"Train data saved to {train_path}")
        print(f"Validation data saved to {val_path}")
        print(f"Test data saved to {test_path}")
    
    return train_path, val_path, test_path, label_to_index

def load_data(train_path, val_path, test_path):
    # Load the split data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    numerical_features = train_df.columns[:-1]  # All columns except the target

    X_train = train_df[numerical_features].values
    y_train = train_df['track_genre_encoded'].values
    X_val = val_df[numerical_features].values
    y_val = val_df['track_genre_encoded'].values
    X_test = test_df[numerical_features].values
    y_test = test_df['track_genre_encoded'].values

    return X_train, y_train, X_val, y_val, X_test, y_test

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

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return self._most_common(k_nearest_labels)

    def _compute_distances(self, x):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(self.X_train - x, axis=1)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.distance_metric == 'cosine':
            return np.array([self._cosine_distance(x, x_train) for x_train in self.X_train])
        else:
            raise ValueError("Unsupported distance metric")

    def _cosine_distance(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2)) if (norm_x1 * norm_x2) != 0 else 1

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

# Main script
train_path, val_path, test_path, label_to_index = load_and_prepare_data()
X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_path, val_path, test_path)

# Initialize and train KNN model
knn = KNN(k=20, distance_metric='euclidean')
knn.fit(X_train, y_train)

# Predict and decode predictions
y_pred = knn.predict(X_test)
y_pred_decoded = decode_labels(y_pred, label_to_index)

# Calculate metrics
metrics = Metrics(y_test, y_pred_decoded)
print(f"Accuracy: {metrics.accuracy()}")
print(f"Precision: {metrics.precision()}")
print(f"Recall: {metrics.recall()}")
print(f"F1 Score: {metrics.f1_score()}")
