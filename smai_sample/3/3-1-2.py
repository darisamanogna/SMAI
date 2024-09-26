import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # For saving and loading the model

class PolynomialRegression:
    def __init__(self, degree=1, regularization=0.0, learning_rate=0.01, iterations=1000):
        self.degree = degree
        self.lambda_ = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta = None
    
    def _poly_features(self, X):
        """ Generate polynomial features up to the specified degree. """
        poly_features = np.hstack([X**i for i in range(self.degree + 1)])
        return poly_features
    
    def fit(self, X, Y):
        X_poly = self._poly_features(X)
        n_samples, n_features = X_poly.shape
        
        # Initialize coefficients (beta) to zeros
        self.beta = np.zeros(n_features)
        
        # Gradient Descent
        for _ in range(self.iterations):
            predictions = X_poly @ self.beta
            errors = predictions - Y
            gradient = (2 / n_samples) * (X_poly.T @ errors) + 2 * self.lambda_ * self.beta
            gradient[0] -= 2 * self.lambda_ * self.beta[0]  # Do not regularize intercept
            self.beta -= self.learning_rate * gradient
    
    def predict(self, X):
        X_poly = self._poly_features(X)
        return X_poly @ self.beta

    def mean_squared_error(self, Y_true, Y_pred):
        return np.mean((Y_true - Y_pred) ** 2)

    def variance(self, Y):
        return np.var(Y)
    
    def standard_deviation(self, Y):
        return np.std(Y)

# Load and prepare data
data = pd.read_csv('linreg.csv')
X = data['x'].values.reshape(-1, 1)
Y = data['y'].values

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Split the data into 80:10:10
train_size = int(0.8 * X.shape[0])
val_size = int(0.1 * X.shape[0])

X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]

# Define degrees to test
degrees = [1, 2, 3, 4, 5]
best_degree = None
best_mse = float('inf')
best_model = None

for degree in degrees:
    model = PolynomialRegression(degree=degree, learning_rate=0.01, iterations=10000)
    model.fit(X_train, Y_train)
    
    # Evaluate on train, validation, and test sets
    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)
    Y_test_pred = model.predict(X_test)

    train_mse = model.mean_squared_error(Y_train, Y_train_pred)
    val_mse = model.mean_squared_error(Y_val, Y_val_pred)
    test_mse = model.mean_squared_error(Y_test, Y_test_pred)
    
    # Print the metrics for the current degree
    print(f"Degree: {degree}")
    print(f"Train MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")
    print(f"Test MSE: {test_mse}")
    print("----------------------------")
    
    if test_mse < best_mse:
        best_mse = test_mse
        best_degree = degree
        best_model = model

# Print the best model results
print("\nBest Polynomial Degree: ", best_degree)
print(f"Train MSE for Best Model: {best_model.mean_squared_error(Y_train, best_model.predict(X_train))}")
print(f"Validation MSE for Best Model: {best_model.mean_squared_error(Y_val, best_model.predict(X_val))}")
print(f"Test MSE for Best Model: {best_model.mean_squared_error(Y_test, best_model.predict(X_test))}")

# Save the best model's parameters
joblib.dump(best_model, 'best_polynomial_regression_model.pkl')

# Plot the data and the best model
plt.scatter(X_train, Y_train, color='blue', label='Train')
plt.scatter(X_val, Y_val, color='orange', label='Validation')
plt.scatter(X_test, Y_test, color='green', label='Test')

# Plot the regression line
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
Y_range_pred_best = best_model.predict(X_range)
plt.plot(X_range, Y_range_pred_best, color='red', linestyle='--', label=f'Best Model (Degree={best_degree})')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression with Data Splits')
plt.legend()
plt.show()
