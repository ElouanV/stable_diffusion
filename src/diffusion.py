
# Simulate forward diffusion for N steps.
def forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt):
    """x0: initial sample value, scalar
    noise_strength_fn: function of time, outputs scalar noise strength
    t0: initial time
    nsteps: number of diffusion steps
    dt: time step size
    """

    # Initialize trajectory
    x = np.zeros(nsteps + 1); x[0] = x0
    t = t0 + np.arange(nsteps + 1 ) *dt

    # Perform many Euler-Maruyama time steps
    for i in range(nsteps):
        noise_strength = noise_strength_fn(t[i])
        ############ YOUR CODE HERE (2 lines)
        random_normal = ...
        x[ i +1] = ...
        #####################################
    return x, t


# Example noise strength function: always equal to 1
def noise_strength_constant(t):
    return 1