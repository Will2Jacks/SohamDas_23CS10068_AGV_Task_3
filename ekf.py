import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta
        self.state_dim = len(mean)

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark observation.

        Args:
            env: Environment class instance.
            u: Action vector.
            z: Landmark observation vector.
            marker_id: ID of the observed landmark.

        Returns:
            Updated state mean and covariance.
        """
        G = env.G(self.mu, u)  # Jacobian of the dynamics with respect to the state
        self.mu = env.forward(self.mu, u)  # Predicted state mean
        alpha = env.noise_from_motion(u, self.alphas)  # Covariance matrix for noisy action
        self.sigma = np.dot(np.dot(G, self.sigma), G.T) + alpha  # Predicted state covariance

        # Update step
        H = env.H(self.mu, marker_id)  # Jacobian of the observation with respect to the state
        z_pred = env.observe(self.mu, marker_id)  # Predicted measurement
        y = z - z_pred  # Innovation
        S = np.dot(np.dot(H, self.sigma), H.T) + self.beta  # Innovation covariance
        K = np.dot(np.dot(self.sigma, H.T), np.linalg.inv(S))  # Kalman gain
        self.mu = self.mu + np.dot(K, y)  # Updated state mean
        self.sigma = np.dot((np.eye(self.state_dim) - np.dot(K, H)), self.sigma)  # Updated state covariance

        return self.mu, self.sigma 