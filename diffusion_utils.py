import jax
import jax.numpy as jnp
import jax.random as random

class SimpleDiffusion:
    def __init__(self, timesteps=200):
        self.timesteps = timesteps
        self.initialize()
    def initialize(self):
        self.beta = jnp.linspace(10**(-4), 0.02, self.timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1 - self.alpha_bar)
def forward_process(key, x_0,  sd:SimpleDiffusion, timestep):
    noise = random.normal(key, x_0.shape)
    reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sd.sqrt_alpha_bar, timestep), (-1, 1))
    reshaped_sqrt_one_minus_alpha_bar_t = jnp.reshape(jnp.take(sd.sqrt_one_minus_alpha_bar, timestep), (-1, 1))
    noisy_image = reshaped_sqrt_alpha_bar_t * x_0 + reshaped_sqrt_one_minus_alpha_bar_t * noise
    return noisy_image

