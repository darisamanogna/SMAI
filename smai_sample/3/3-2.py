import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, regularization=0.0, learning_rate=0.01, iterations=1000):
        self.lambda_ = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta = None

    def fit(self, X, Y):
        # Add a column of ones for the intercept term
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        n_samples, n_features = X.shape
        
        # Initialize coefficients (beta) to zeros
        self.beta = np.zeros(n_features)
        
        # Gradient Descent
        for _ in range(self.iterations):
            predictions = X @ self.beta
            errors = predictions - Y
            gradient = (2 / n_samples) * (X.T @ errors) + 2 * self.lambda_ * self.beta
            gradient[0] -= 2 * self.lambda_ * self.beta[0]  # Do not regularize intercept
            self.beta -= self.learning_rate * gradient

    def predict(self, X):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return X @ self.beta

    def mean_squared_error(self, Y_true, Y_pred):
        return np.mean((Y_true - Y_pred) ** 2)

    def variance(self, Y):
        return np.var(Y)

    def standard_deviation(self, Y):
        return np.std(Y)

# Load the data
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

# Experiment with different learning rates
learning_rates = [0.01, 0.001, 0.0001]
best_mse = float('inf')
best_lr = None
best_model = None

# Store models for plotting
models = {}

for lr in learning_rates:
    model = LinearRegression(learning_rate=lr, iterations=10000)
    model.fit(X_train, Y_train)
    
    # Evaluate on train, validation, and test sets
    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)
    Y_test_pred = model.predict(X_test)

    train_mse = model.mean_squared_error(Y_train, Y_train_pred)
    val_mse = model.mean_squared_error(Y_val, Y_val_pred)
    test_mse = model.mean_squared_error(Y_test, Y_test_pred)
    
    # Print the metrics for the current learning rate
    print(f"Learning Rate: {lr}")
    print(f"Train MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")
    print(f"Test MSE: {test_mse}")
    print("----------------------------")
    
    # Store the model for plotting
    models[lr] = model
    
    if val_mse < best_mse:
        best_mse = val_mse
        best_lr = lr
        best_model = model

# Print the best model results
print("\nBest Learning Rate: ", best_lr)
print(f"Train MSE for Best Model: {best_model.mean_squared_error(Y_train, best_model.predict(X_train))}")
print(f"Validation MSE for Best Model: {best_model.mean_squared_error(Y_val, best_model.predict(X_val))}")
print(f"Test MSE for Best Model: {best_model.mean_squared_error(Y_test, best_model.predict(X_test))}")

# Calculate and print variance and standard deviation for the best model
Y_train_pred_best = best_model.predict(X_train)
Y_test_pred_best = best_model.predict(X_test)

train_var_best = best_model.variance(Y_train)
test_var_best = best_model.variance(Y_test)
train_std_best = best_model.standard_deviation(Y_train)
test_std_best = best_model.standard_deviation(Y_test)

print("\nBest Model Metrics:")
print(f"Train Variance: {train_var_best}")
print(f"Test Variance: {test_var_best}")
print(f"Train Standard Deviation: {train_std_best}")
print(f"Test Standard Deviation: {test_std_best}")

# Plot the data
plt.scatter(X_train, Y_train, color='blue', label='Train')
plt.scatter(X_val, Y_val, color='orange', label='Validation')
plt.scatter(X_test, Y_test, color='green', label='Test')

# Plot the regression lines for all learning rates
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
for lr, model in models.items():
    Y_range_pred = model.predict(X_range)
    plt.plot(X_range, Y_range_pred, label=f'Model (LR={lr})')

# Highlight the best model
Y_range_pred_best = best_model.predict(X_range)
plt.plot(X_range, Y_range_pred_best, color='red', linestyle='--', label=f'Best Model (LR={best_lr})')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Data Splits and Learning Rates')
plt.legend()
plt.show()
