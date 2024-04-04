import numpy as np

from utils import minimized_angle




class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta, resampling_threshold=0.5, control_noise=0.1):
        self.alphas = alphas
        self.beta = beta
        self.resampling_threshold = resampling_threshold
        self.control_noise = control_noise

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.random.multivariate_normal(self._init_mean.ravel(), self._init_cov, size=self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        # Prediction step
        self.predict(u)

        # Measurement update
        likelihoods = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            particle = self.particles[i].reshape((-1, 1))
            z_expected = env.observe(particle, marker_id)
            innovation = z - z_expected
            likelihoods[i] = env.likelihood(innovation, self.beta)

        # Regularization to prevent zero or near-zero weights
        likelihoods += 1e-12

        # Normalize weights
        self.weights = likelihoods / likelihoods.sum()

        # Effective Sample Size (ESS) Check
        ess = 1. / np.sum(np.square(self.weights))
        if ess < self.resampling_threshold * self.num_particles:
            self.resample()

        # Compute weighted mean
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def predict(self, u):
        # Propagate particles based on control input (action)
        num_particles = len(self.weights)
        noise = np.random.multivariate_normal(np.zeros(3), self.control_noise * np.eye(3), size=num_particles)
        self.particles += u + noise

    def resample(self):
        num_particles = len(self.weights)
        
        # Systematic resampling
        cumulative_weights = np.cumsum(self.weights)
        step = cumulative_weights[-1] / num_particles
        r = np.random.uniform(0, step)
        indexes = np.arange(num_particles) + r
        indexes %= cumulative_weights[-1]
        indexes = np.searchsorted(cumulative_weights, indexes)
        self.particles = self.particles[indexes]
        
        # Add noise to particles
        noise = np.random.multivariate_normal(np.zeros(3), 0.01 * np.eye(3), size=num_particles)
        self.particles += noise

        # Reset weights
        self.weights = np.ones(num_particles) / num_particles

    def mean_and_variance(self, particles):
        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(np.sum(np.sin(particles[:, 2])), np.sum(np.cos(particles[:, 2])))
        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        return mean.reshape((-1, 1)), cov
