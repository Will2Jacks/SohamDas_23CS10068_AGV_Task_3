Note: Only those functions have been included that have been modified based on the existing code.
The functions that have been modified are soccerfield ekf and pf

The jacobian of the dynamics with respect to the state, the jacobian of the dynamics with respect to the control and the jacobian of the observation with respect to the state are calculated.
In all three of these the determinant of the jacobian matrix has been returned with values calculated on the basis of the equations provided.

Calculating Extended Kalman Filtering

self.mu is updated to the predicted state mean using the moving forward dynamics model
alpha represents the covariance matrix for the noise in the action, calculated using env.noise_from_motion.
self.sigma is updated to the predicted state covariance using the standard Kalman filter prediction equation.

z_pred is the predicted measurement based on the current state estimate and the observation model.
y represents the innovation or the difference between the actual measurement z and the predicted measurement z_pred.
S is the innovation covariance matrix, consisting of the predicted measurement uncertainty and some additional noise term self.beta.
K is the Kalman gain, calculated using the standard formula.
self.mu is updated using the Kalman gain and innovation y.
self.sigma is updated to incorporate the new information gained from the observation.

Calculating Particle Filtering
The update method performs the measurement update step of the particle filter.
It calculates the likelihood of each particle given the current measurement (z) and updates the weights accordingly.
After updating the weights, it checks the Effective Sample Size (ESS) to determine if resampling is necessary.

If the ESS falls below a certain threshold (resampling_threshold * num_particles), indicating that the particles are degenerate, the resample method is called.
Resampling is performed using systematic resampling, where particles are selected based on their weights. Noise is added to the resampled particles to maintain diversity.
The weights are reset to uniform after resampling.
