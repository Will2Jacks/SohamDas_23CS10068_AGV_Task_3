Note: Only those functions have been included that have been modified based on the existing code.
The functions that have been modified are soccerfield ekf and pf

The jacobian of the dynamics with respect to the state, the jacobian of the dynamics with respect to the control and the jacobian of the observation with respect to the state are calculated.
In all three of these the determinant of the jacobian matrix has been returned with values calculated on the basis of the equations provided.

