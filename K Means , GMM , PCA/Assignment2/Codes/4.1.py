import numpy as np

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components  # Number of Gaussian components
        self.max_iter = max_iter  # Maximum number of iterations for EM
        self.tol = tol  # Convergence threshold
        self.means = None  # Means of the Gaussian components
        self.covariances = None  # Covariances of the Gaussian components
        self.weights = None  # Weights of the Gaussian components

    # Helper function to calculate the multivariate normal distribution
    def _multivariate_gaussian(self, X, mean, cov):
        n_samples, n_features = X.shape
        cov_inv = np.linalg.inv(cov)
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        norm_factor = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(cov))
        return exp_term / norm_factor

    def fit(self, X):
        n_samples, n_features = X.shape
        # Initialize means, covariances, and weights
        self.means = X[np.random.choice(n_samples, self.n_components, False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

        log_likelihood_old = 0
        for _ in range(self.max_iter):
            # E-Step: Compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-Step: Update parameters
            self._m_step(X, responsibilities)
            
            # Check convergence
            log_likelihood_new = self._compute_log_likelihood(X)
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

    def _e_step(self, X):
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            resp[:, i] = self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])

        resp_sum = resp.sum(axis=1)[:, np.newaxis]
        return resp / resp_sum

    def _m_step(self, X, responsibilities):
        nk = responsibilities.sum(axis=0)  # Sum of responsibilities for each component
        self.weights = nk / X.shape[0]  # Update weights
        self.means = np.dot(responsibilities.T, X) / nk[:, np.newaxis]  # Update means
        
        # Update covariances
        n_features = X.shape[1]
        self.covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / nk[k]

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for i in range(self.n_components):
            log_likelihood += self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
        return np.sum(np.log(log_likelihood))

    def getParams(self):
        return self.means, self.covariances, self.weights

    def getMembership(self, X):
        responsibilities = self._e_step(X)
        return responsibilities

    def getLikelihood(self, X):
        return self._compute_log_likelihood(X)


