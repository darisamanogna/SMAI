# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Load the data
# data = pd.read_csv('linreg.csv')
# X = data['x'].values.reshape(-1, 1)
# Y = data['y'].values

# # Shuffle the data
# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# Y = Y[indices]

# # Split the data into 80:10:10
# train_size = int(0.8 * X.shape[0])
# val_size = int(0.1 * X.shape[0])

# X_train, Y_train = X[:train_size], Y[:train_size]
# X_val, Y_val = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
# X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]

# class PolynomialRegression:
#     def __init__(self, degree=1, regularization=0.0, learning_rate=0.01, iterations=1000):
#         self.degree = degree
#         self.lambda_ = regularization
#         self.learning_rate = learning_rate
#         self.iterations = iterations
#         self.beta = None
    
#     def _poly_features(self, X):
#         """Generate polynomial features up to the specified degree."""
#         poly_features = np.hstack([X**i for i in range(self.degree + 1)])
#         return poly_features
    
#     def fit(self, X, Y):
#         X_poly = self._poly_features(X)
#         n_samples, n_features = X_poly.shape
        
#         # Initialize coefficients (beta) to zeros
#         self.beta = np.zeros(n_features)
        
#         # Gradient Descent
#         for _ in range(self.iterations):
#             predictions = X_poly @ self.beta
#             errors = predictions - Y
#             gradient = (2 / n_samples) * (X_poly.T @ errors) + 2 * self.lambda_ * self.beta
#             gradient[0] -= 2 * self.lambda_ * self.beta[0]  # Do not regularize intercept
#             self.beta -= self.learning_rate * gradient
    
#     def predict(self, X):
#         X_poly = self._poly_features(X)
#         return X_poly @ self.beta

#     def mean_squared_error(self, Y_true, Y_pred):
#         return np.mean((Y_true - Y_pred) ** 2)

# # Set up the figure and axis
# fig, ax = plt.subplots()

# # Initialize empty list to store polynomial degree
# degrees = [1, 2, 3, 4, 5]
# models = [PolynomialRegression(degree=d, learning_rate=0.01, iterations=10000) for d in degrees]

# def update(frame):
#     ax.clear()
#     degree = degrees[frame]
#     model = models[frame]
#     model.fit(X_train, Y_train)
    
#     X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#     Y_range_pred = model.predict(X_range)
    
#     ax.scatter(X_train, Y_train, color='blue', label='Train')
#     ax.scatter(X_val, Y_val, color='orange', label='Validation')
#     ax.scatter(X_test, Y_test, color='green', label='Test')
#     ax.plot(X_range, Y_range_pred, color='red', label=f'Poly Degree {degree}')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'Polynomial Regression (Degree {degree})')
#     ax.legend()

# ani = animation.FuncAnimation(fig, update, frames=len(degrees), repeat=True)

# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

class PolynomialRegression:
    def __init__(self, degree=1, regularization=0.0, learning_rate=0.01, iterations=1000):
        self.degree = degree
        self.lambda_ = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta = None
    
    def _poly_features(self, X):
        """Generate polynomial features up to the specified degree."""
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

# Set up the figure and axis
fig, ax = plt.subplots()

# Initialize empty list to store polynomial degree
degrees = [1, 2, 3, 4, 5]
models = [PolynomialRegression(degree=d, learning_rate=0.01, iterations=10000) for d in degrees]

def update(frame):
    ax.clear()
    degree = degrees[frame]
    model = models[frame]
    model.fit(X_train, Y_train)
    
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    Y_range_pred = model.predict(X_range)
    
    ax.scatter(X_train, Y_train, color='blue', label='Train')
    ax.scatter(X_val, Y_val, color='orange', label='Validation')
    ax.scatter(X_test, Y_test, color='green', label='Test')
    ax.plot(X_range, Y_range_pred, color='red', label=f'Poly Degree {degree}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Polynomial Regression (Degree {degree})')
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(degrees), repeat=True)

# Save the animation as a GIF
ani.save('figures/animation.gif', writer='pillow', fps=2)
