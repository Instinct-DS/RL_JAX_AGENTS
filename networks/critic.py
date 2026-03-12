import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Optional

class Q_network(nn.Module):
    state_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, state, action):
        # Activation map
        activation_map = {
            "relu": nn.relu,
            "leakyrelu": nn.leaky_relu,
            "gelu": nn.gelu,
            "selu": nn.selu,
            "silu": nn.silu,
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
        }
        
        # Handle mish if not in nn (older flax versions)
        act_fn = activation_map.get(self.activation, nn.relu)
        x = jnp.concatenate([state, action], axis=-1)

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(features=hidden_size)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)

        x = nn.Dense(features=1)(x)
        return x.squeeze(-1) # Match pytorch shape (batch,)
    
class CombinedCritics(nn.Module):
    state_dim: int
    action_dim: int
    n_critics: int
    critic_kwargs: dict

    @nn.compact
    def __call__(self, state, action):
        # We use vmap to create n_critics independent networks
        # variable_axes={'params': 0} indicates the 0th axis of parameters corresponds to the critic index
        # split_rngs={'params': True} ensures each critic gets different initialization
        VectorizedCritic = nn.vmap(
            Q_network,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics
        )
        
        critics = VectorizedCritic(
            state_dim=self.state_dim, 
            action_dim=self.action_dim, 
            **self.critic_kwargs
        )
        
        # Returns shape: (n_critics, batch)
        return critics(state, action)
    
class V_network(nn.Module):
    state_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, state):
        # Activation map
        activation_map = {
            "relu": nn.relu,
            "leakyrelu": nn.leaky_relu,
            "gelu": nn.gelu,
            "selu": nn.selu,
            "silu": nn.silu,
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
        }
        
        # Handle mish if not in nn (older flax versions)
        act_fn = activation_map.get(self.activation, nn.relu)
        x = state

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(features=hidden_size)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)

        x = nn.Dense(features=1)(x)
        return x.squeeze(-1) # Match pytorch shape (batch,)
    
class CombinedBaselines(nn.Module):
    state_dim: int
    n_critics: int
    critic_kwargs: dict

    @nn.compact
    def __call__(self, state):
        # We use vmap to create n_critics independent networks
        # variable_axes={'params': 0} indicates the 0th axis of parameters corresponds to the critic index
        # split_rngs={'params': True} ensures each critic gets different initialization
        VectorizedCritic = nn.vmap(
            V_network,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics
        )
        
        critics = VectorizedCritic(
            state_dim=self.state_dim, 
            **self.critic_kwargs
        )
        
        # Returns shape: (n_critics, batch)
        return critics(state)
    
def init_critic_from_baseline(
    baseline_params,      # params of CombinedBaselines
    critic_params,        # params of CombinedCritics (used as shape template)
    state_dim: int,
    action_dim: int,
    noise_scale: float = 1e-4,
    rng: jax.Array = None,
):
    """
    Copy state weights from baseline layer 1 into critic layer 1[:state_dim, :].
    The action portion critic layer 1[state_dim:, :] is filled with near-zero noise.
    All other layers (hidden->hidden) are copied directly.
    Biases are copied for all layers.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def transfer_single(b_params, c_params, rng):
        new_c = {}

        for layer_key in c_params.keys():
            b_layer = b_params[layer_key]
            c_layer = c_params[layer_key]

            if layer_key == "Dense_0":
                # Baseline kernel: (state_dim, hidden)
                # Critic kernel needs: (state_dim + action_dim, hidden)
                state_kernel = b_layer["kernel"]  # (state_dim, hidden)
                hidden_size = state_kernel.shape[-1]

                rng, noise_key = jax.random.split(rng)
                action_kernel = noise_scale * jax.random.normal(
                    noise_key, shape=(action_dim, hidden_size)
                )

                new_c[layer_key] = {
                    "kernel": jnp.concatenate([state_kernel, action_kernel], axis=0),
                    "bias": b_layer["bias"],
                }
            else:
                # Deeper layers: identical shape, copy directly
                new_c[layer_key] = {
                    "kernel": b_layer["kernel"],
                    "bias": b_layer["bias"],
                }

        # Handle LayerNorm params if present
        for layer_key in c_params.keys():
            if layer_key.startswith("LayerNorm"):
                new_c[layer_key] = b_params.get(layer_key, c_params[layer_key])

        return new_c

    # Unwrap vmap module wrapper keys
    critic_inner_key   = list(critic_params.keys())[0]    # e.g. "VmapQ_network_0"
    baseline_inner_key = list(baseline_params.keys())[0]  # e.g. "VmapV_network_0"

    c_inner = critic_params[critic_inner_key]
    b_inner = baseline_params[baseline_inner_key]

    # Split rng across n_critics
    n_critics = jax.tree_util.tree_leaves(c_inner)[0].shape[0]
    rngs = jax.random.split(rng, n_critics)

    transfer_vmapped = jax.vmap(transfer_single)
    new_c_inner = transfer_vmapped(b_inner, c_inner, rngs)

    return {critic_inner_key: new_c_inner}