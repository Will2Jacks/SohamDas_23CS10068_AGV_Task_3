if np.any(np.isnan(likelihoods)) or np.any(likelihoods <= 0):
        #     # Reset weights to uniform distribution
        #     self.weights = np.ones(self.num_particles) / self.num_particles
        # else:
        #     # Normalize weights