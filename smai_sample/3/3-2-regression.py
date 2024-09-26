import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, degree=1, regularization_param=0.0):
        self.degree = degree
        self.regularization_param = regularization_param
        self.coefficients = None

    def _create_polynomial_features(self, X):
        """Creates polynomial features for the input data."""
        X_poly = np.vander(X, N=self.degree + 1, increasing=True)
        return X_poly

    def fit(self, X, y, learning_rate=0.001, epochs=1000):
        X_poly = self._create_polynomial_features(X)
        n_samples, n_features = X_poly.shape
        
        # Initialize coefficients (theta) randomly
        self.coefficients = np.random.randn(n_features)
        
        # Gradient Descent
        for epoch in range(epochs):
            predictions = X_poly @ self.coefficients
            errors = predictions - y
            gradient = (2/n_samples) * (X_poly.T @ errors) + self.regularization_param * np.sign(self.coefficients)
            gradient[0] -= self.regularization_param * np.sign(self.coefficients[0])  # Do not regularize intercept
            self.coefficients -= learning_rate * gradient
    
    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.coefficients

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def std_dev(self, y_pred):
        return np.std(y_pred)
    
    def variance(self, y_pred):
        return np.var(y_pred)

    def save_coefficients(self, filename):
        np.save(filename, self.coefficients)
    
    def load_coefficients(self, filename):
        self.coefficients = np.load(filename)

# Data processing, shuffling, and splitting
def preprocess_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    np.random.shuffle(data)
    X = data[:, 0]
    y = data[:, 1]
    return X, y

def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Visualization function
def plot_results(X_train, y_train, X_val, y_val, X_test, y_test, model, degree):
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_val, y_val, color='orange', label='Validation data')
    plt.scatter(X_test, y_test, color='green', label='Test data')
    
    x_range = np.linspace(min(X_train), max(X_train), 100)
    plt.plot(x_range, model.predict(x_range), color='red', label=f'Degree {degree} fit')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Load and preprocess data
X, y = preprocess_data('regularisation.csv')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Initialize and fit the model for Degree 1
model = LinearRegression(degree=1, regularization_param=0.0)
model.fit(X_train, y_train, learning_rate=0.01, epochs=1000)

# Predict and calculate metrics
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = model.mse(y_train, y_train_pred)
mse_test = model.mse(y_test, y_test_pred)
std_dev_train = model.std_dev(y_train_pred)
std_dev_test = model.std_dev(y_test_pred)
variance_train = model.variance(y_train_pred)
variance_test = model.variance(y_test_pred)

# Print the metrics on separate lines
print(f'MSE (Train): {mse_train}')
print(f'MSE (Test): {mse_test}')
print(f'Standard Deviation (Train): {std_dev_train}')
print(f'Standard Deviation (Test): {std_dev_test}')
print(f'Variance (Train): {variance_train}')
print(f'Variance (Test): {variance_test}')

# Visualize the results for Degree 1
plot_results(X_train, y_train, X_val, y_val, X_test, y_test, model, degree=1)
