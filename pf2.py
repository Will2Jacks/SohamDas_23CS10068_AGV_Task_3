import numpy as np
from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov
            )
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        likelihoods = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            particle = self.particles[i].reshape((-1, 1))
            z_expected = env.observe(particle, marker_id)
            innovation = z - z_expected
            beta = env.beta
            likelihoods[i] = env.likelihood(innovation, beta)

        # Normalize weights and avoid division by zero
        total_likelihood = np.sum(likelihoods)
        if total_likelihood == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = likelihoods / total_likelihood

        # Selective resampling based on effective number of particles or best particle weight
        effective_particles = 1.0 / np.sum(self.weights**2)
        best_particle_weight = np.max(self.weights)
        if effective_particles < self.num_particles / 2 or best_particle_weight > 0.95:
            self.resample()

        # Compute mean and covariance
        mean, cov = self.mean_and_variance(self.particles)

        # Add regularization to covariance
        cov += 5000 * np.eye(cov.shape[0])

        return mean, cov

    def resample(self):
        # Option 1: Selective resampling based on weights
        # if effective_particles < self.num_particles / 2 or best_particle_weight > 0.95:

        # Option 2: Adaptive resampling
        cumsum_weights = np.cumsum(self.weights)
        random_values = np.random.rand(self.num_particles)
        indices = np.searchsorted(cumsum_weights, random_values)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def mean_and_variance(self, particles):
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(), np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
