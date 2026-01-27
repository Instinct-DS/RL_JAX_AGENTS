from typing import Any, Optional

import distrax
import jax
import jax.numpy as jnp

# Inspired by
# https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# but modified to only compute a mode.


class TanhTransformedDistribution(distrax.Transformed):
    def __init__(self, distribution: distrax.Distribution):
        # distribution: MultivariateNormalDiag with event_shape = (action_dim,)
        bijector = distrax.Block(distrax.Tanh(), ndims=1)
        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
    
from typing import Tuple

EPS = 1e-6

class OwnTanhTransform:
    def __init__(self, base_distribution: distrax.Distribution):
        self.base = base_distribution

    def sample(self, seed):
        u = self.base.sample(seed=seed)
        u = jnp.clip(u, -20.0, 20.0)
        return jnp.tanh(u)

    def sample_and_log_prob(self, seed):
        u = self.base.sample(seed=seed)
        u = jnp.clip(u, -20.0, 20.0)
        a = jnp.tanh(u)

        pre_sum = self.base.log_prob(u)
        log_det = jnp.sum(
            2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), 
            axis=-1
        )

        log_prob = pre_sum - log_det
        return a, log_prob

    def log_prob(self, actions):
        a = jnp.clip(actions, -1 + EPS, 1 - EPS)
        u = 0.5 * (jnp.log1p(a) - jnp.log1p(-a))
        u = jnp.clip(u, -20.0, 20.0)

        pre_sum = self.base.log_prob(u)
        log_det = jnp.sum(
            2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), 
            axis=-1
        )

        return pre_sum - log_det

    def mode(self):
        return jnp.tanh(self.base.loc)

import flax.linen as nn
class SafeTanhTransform(distrax.Tanh):
    """A numerically stable tanh transform with gradient clipping."""
    
    def __init__(self, base_distribution: distrax.Distribution):
        super().__init__(base_distribution)
    
    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        # Add safety clipping to prevent extreme values
        x = jnp.clip(x, -20.0, 20.0)
        return jnp.tanh(x)
    
    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # Use stable inverse tanh formula
        y = jnp.clip(y, -1.0 + EPS, 1.0 - EPS)
        return jnp.arctanh(y)
    
    def _forward_log_det_jacobian(self, x: jnp.ndarray) -> jnp.ndarray:
        # More stable calculation of log determinant
        x = jnp.clip(x, -20.0, 20.0)
        return 2. * (jnp.log(2.) - x - nn.softplus(-2. * x))
    
    def sample_and_log_prob(self, seed, **kwargs):
        # Override to ensure numerical stability
        x = self.distribution.sample(seed=seed, **kwargs)
        x = jnp.clip(x, -20.0, 20.0)
        y = self._forward(x)
        
        # Compute log probability with stability
        log_prob_base = self.distribution.log_prob(x)
        log_det = self._forward_log_det_jacobian(x)
        
        # Sum over action dimensions
        log_det = jnp.sum(log_det, axis=-1)
        log_prob = log_prob_base - log_det
        
        return y, log_prob

