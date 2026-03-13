import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax.linen as nn
import optax
import numpy as np
import random
from time import time
from tqdm import tqdm
from collections import deque, namedtuple
import mlflow

from common.replaybuffer import ReplayBuffer
from common.logger import MLFlowLogger
from common.utils import load_demo_trajectories, load_demo_trajectories_parallel

from networks.critic import Q_network, CombinedCritics, V_network, CombinedBaselines, init_critic_from_baseline
from networks.actor import DeterministicPolicy, TanhGaussianPolicy
from flax.training.train_state import TrainState

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'termination', 'truncation', 'next_state'])

class AlphaModule(nn.Module):
    ent_init: float
    
    @nn.compact
    def __call__(self):
        log_alpha = self.param('log_alpha', init_fn=lambda key: jnp.full((), jnp.log(self.ent_init)))
        return log_alpha

# Define the functions required for SAC+ML offlinetraining
def _update_critic_sacml(
        critic_state, target_critic_params, actor_state, alpha_state,
        v_critic_state, v_target_critic_params,
        batch, rng, gamma 
):
    # Split the keys
    rng, key_target_policy = jax.random.split(rng,2)

    # Get the target actions from the actor
    next_actions, next_log_probs = actor_state.apply_fn(
        actor_state.params, batch["next_observations"], key_target_policy, method=TanhGaussianPolicy.sample
    ) 

    # Get the Q-targets from the target critics
    target_qs = critic_state.apply_fn(target_critic_params, batch["next_observations"], next_actions)
    min_target_q = jnp.min(target_qs, axis=0)

    # Get the alpha
    log_alpha = alpha_state.apply_fn(alpha_state.params)
    alpha = jnp.exp(log_alpha)

    # Compute Bellman Target
    q_backup = batch["rewards"] + (1 - batch["terminations"]) * (gamma) * (min_target_q - alpha * next_log_probs)

    # Calculate Critic Loss
    def critic_loss_fn(p):
        current_qs = critic_state.apply_fn(p, batch["observations"], batch["actions"])
        loss = jnp.mean(jnp.square(current_qs - q_backup[None, :]))
        return loss
    
    critic_grad_fn = jax.value_and_grad(critic_loss_fn)
    critic_loss, critic_grads = critic_grad_fn(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=critic_grads)

    # Get the V-targets from the target critics
    target_vs =v_critic_state.apply_fn(v_target_critic_params, batch["next_observations"])
    min_target_v = jnp.min(target_vs, axis=0)

    # Compute Bellman Target
    target_log_probs = actor_state.apply_fn(actor_state.params, batch["next_observations"], next_actions, method=TanhGaussianPolicy.log_probs) 
    v_backup = batch["rewards"] + (1 - batch["terminations"]) * (gamma) * (min_target_v + target_log_probs - alpha*target_log_probs)

    # Calculate Critic Loss
    def v_critic_loss_fn(p):
        current_vs = v_critic_state.apply_fn(p, batch["observations"]) + actor_state.apply_fn(actor_state.params, batch["observations"], batch["actions"], method=TanhGaussianPolicy.log_probs) 
        loss = jnp.mean(jnp.square(current_vs - v_backup[None, :]))
        return loss
    
    v_critic_grad_fn = jax.value_and_grad(v_critic_loss_fn)
    v_critic_loss, v_critic_grads = v_critic_grad_fn(v_critic_state.params)
    new_v_critic_state = v_critic_state.apply_gradients(grads=v_critic_grads)


    return new_critic_state, critic_loss, new_v_critic_state, v_critic_loss

def _update_actor_and_alpha_sacml(
        actor_state, critics_state, alpha_state,
        batch, rng, target_entropy, omega
    ):
        states = batch["observations"]
        actions = batch["actions"]
        rng, key_actor = jax.random.split(rng)
        
        # Get Alpha
        log_alpha = alpha_state.apply_fn(alpha_state.params)
        alpha = jnp.exp(log_alpha)

        # Update Actor
        def actor_loss_fn(p):
            # Sample actions from the policy
            curr_actions, curr_log_probs = actor_state.apply_fn(p, states, key_actor, method=TanhGaussianPolicy.sample)
            
            # Get Q-values from the *updated* critic
            qs_pi = critics_state.apply_fn(critics_state.params, states, curr_actions)
            q_pi = jnp.min(qs_pi, axis=0)
            
            # Get Q estimation for the action in the batch
            q_mu_pi = critics_state.apply_fn(critics_state.params, states, actions)
            q_mu = jnp.min(q_mu_pi, axis=0)

            lambda_h = jax.lax.stop_gradient(omega / jnp.clip((jnp.mean(jnp.abs(q_mu)) + 1e-8), max=200))
            BC_loss = jnp.mean(actor_state.apply_fn(p, states, actions, method=TanhGaussianPolicy.log_probs))
            
            # Actor loss: alpha * log_pi - Q
            loss = lambda_h * jnp.mean(jax.lax.stop_gradient(alpha) * curr_log_probs - q_pi) - BC_loss
            return loss, curr_log_probs

        # Compute gradients (has_aux=True to keep log_probs for alpha update)
        (actor_loss, log_probs_for_alpha), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        
        # Update Alpha
        def alpha_loss_fn(p):
            cur_log_alpha = alpha_state.apply_fn(p)
            return -(jnp.exp(cur_log_alpha) * jnp.mean(jax.lax.stop_gradient(log_probs_for_alpha + target_entropy)))
            
        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        
        return new_actor_state, new_alpha_state, actor_loss, alpha_loss, alpha

def _clip_beta(
        actor_offline_state, state, action, beta
):
    mu_s, logstd_s = actor_offline_state.apply_fn(actor_offline_state.params, state)
    logp = actor_offline_state.apply_fn(actor_offline_state.params, state, action, method=TanhGaussianPolicy.log_probs)
    logp_minus1 = actor_offline_state.apply_fn(actor_offline_state.params, state, mu_s - beta*jnp.exp(logstd_s), method=TanhGaussianPolicy.log_probs)
    logp_plus1 = actor_offline_state.apply_fn(actor_offline_state.params, state, mu_s + beta*jnp.exp(logstd_s), method=TanhGaussianPolicy.log_probs)
    c_beta = jnp.min(logp_minus1, logp_plus1)

    clip_beta = jnp.logaddexp(logp - c_beta, 0) + c_beta
    return clip_beta

# Define the functions required for SAC+ML offlinetraining
def _update_critic_aca_sac(
        critic_state, target_critic_params, actor_state, alpha_state,
        batch, rng, gamma, actor_offline_state, kappa, beta
):
    # Split the keys
    rng, key_target_policy = jax.random.split(rng,2)

    # Get the target actions from the actor
    next_actions, next_log_probs = actor_state.apply_fn(
        actor_state.params, batch["next_observations"], key_target_policy, method=TanhGaussianPolicy.sample
    ) 

    # Get the Q-targets from the target critics
    target_qs = critic_state.apply_fn(target_critic_params, batch["next_observations"], next_actions)
    min_target_q = jnp.min(target_qs, axis=0)

    # Get the alpha
    log_alpha = alpha_state.apply_fn(alpha_state.params)
    alpha = jnp.exp(log_alpha)

    # Compute Bellman Target
    q_backup = batch["rewards"] + (1 - batch["terminations"]) * (gamma) * ((min_target_q + 
                            kappa * _clip_beta(actor_offline_state, batch["next_observations"], next_actions, beta)) - alpha * next_log_probs)

    # Calculate Critic Loss
    def critic_loss_fn(p):
        current_qs = critic_state.apply_fn(p, batch["observations"], batch["actions"]) + kappa * _clip_beta(actor_offline_state, batch["observations"], batch["actions"], beta)
        loss = jnp.mean(jnp.square(current_qs - q_backup[None, :]))
        return loss
    
    critic_grad_fn = jax.value_and_grad(critic_loss_fn)
    critic_loss, critic_grads = critic_grad_fn(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=critic_grads)

    return new_critic_state, critic_loss

def _update_actor_and_alpha_aca_sac(
        actor_state, critics_state, alpha_state,
        batch, rng, target_entropy, actor_offline_state, kappa, beta
    ):
        states = batch["observations"]
        actions = batch["actions"]
        rng, key_actor = jax.random.split(rng)
        
        # Get Alpha
        log_alpha = alpha_state.apply_fn(alpha_state.params)
        alpha = jnp.exp(log_alpha)

        # Update Actor
        def actor_loss_fn(p):
            # Sample actions from the policy
            curr_actions, curr_log_probs = actor_state.apply_fn(p, states, key_actor, method=TanhGaussianPolicy.sample)
            
            # Get Q-values from the *updated* critic
            qs_pi = critics_state.apply_fn(critics_state.params, states, curr_actions)
            q_pi = jnp.min(qs_pi, axis=0)
            
            # Actor loss: alpha * log_pi - Q
            loss = jnp.mean(jax.lax.stop_gradient(alpha) * curr_log_probs - (q_pi + kappa * _clip_beta(actor_offline_state, states, actions, beta)))
            return loss, curr_log_probs

        # Compute gradients (has_aux=True to keep log_probs for alpha update)
        (actor_loss, log_probs_for_alpha), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        
        # Update Alpha
        def alpha_loss_fn(p):
            cur_log_alpha = alpha_state.apply_fn(p)
            return -jnp.mean(jnp.exp(cur_log_alpha) * (jax.lax.stop_gradient(log_probs_for_alpha) + target_entropy))
            
        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        
        return new_actor_state, new_alpha_state, actor_loss, alpha_loss, alpha


class ACA_Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            tau=0.005,
            gamma=0.99,
            alpha_init=1.0,
            lr=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            n_steps=1,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=1,
            policy_delay_update=1,
            target_entropy=None,
            omega = 100.0,
            kappa = 1.0,
            beta = 15.0,
            seed=0,
            stats_window_size=100,
            n_envs=4,
            n_critics=2,
            m_critics=2,
            logger_name="mlflow",
            policy_kwargs=dict(),
            critic_kwargs=dict(layer_norm=True),
            mlflow_uri = None,
            experiment_name="",
            run_name="",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_steps = n_steps
        self.alpha_init = alpha_init
        assert n_steps == 1, "Support for n_step != 1 is not available!!!"
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.policy_delay_update = policy_delay_update
        self.beta = beta
        self.omega = omega
        self.kappa = kappa
        self.m_critics = m_critics
        self.n_critics = n_critics
        self.logger_name = logger_name
        self.policy_kwargs = policy_kwargs
        self.critic_kwargs = critic_kwargs
        
        # Logging
        self._count_total_gradients_taken = 0
        self._ep_rewards = deque(maxlen=stats_window_size)
        self._ep_lengths = deque(maxlen=stats_window_size)
        self._start_time = time()

        # Random Keys
        self.rng = jax.random.PRNGKey(seed if seed is not None else 0)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize Networks & Optimizers
        self.rng, actor_offline_key, critic_offline_key, v_critic_offline_key, alpha_offline_key = jax.random.split(self.rng, 5)
        self.rng, actor_online_key = jax.random.split(self.rng, 2)

        # 1. Actor
        self.actor_offline_def = TanhGaussianPolicy(state_dim=state_dim, action_dim=action_dim, **policy_kwargs)
        dummy_obs = jnp.zeros((1, state_dim))
        actor_offline_params = self.actor_offline_def.init(actor_offline_key, dummy_obs)
        self.actor_offline_state = TrainState.create(
            apply_fn=self.actor_offline_def.apply,
            params=actor_offline_params,
            tx=optax.adam(lr)
        )

        self.actor_online_def = TanhGaussianPolicy(state_dim=state_dim, action_dim=action_dim, **policy_kwargs)
        dummy_obs = jnp.zeros((1, state_dim))
        actor_online_params = self.actor_online_def.init(actor_online_key, dummy_obs)
        self.actor_online_state = TrainState.create(
            apply_fn=self.actor_online_def.apply,
            params=actor_online_params,
            tx=optax.adam(lr)
        )

        # 2. Critics
        self.critics_def = CombinedCritics(
            state_dim=state_dim, action_dim=action_dim, 
            n_critics=n_critics, critic_kwargs=critic_kwargs
        )
        dummy_action = jnp.zeros((1, action_dim))
        critic_params = self.critics_def.init(critic_offline_key, dummy_obs, dummy_action)
        self.critics_state = TrainState.create(
            apply_fn=self.critics_def.apply,
            params=critic_params,
            tx=optax.adam(lr)
        )
        
        # Target Critics (Just params, no optimizer)
        self.target_critics_params = critic_params

        # Value Critics
        self.v_critics_def = CombinedBaselines(
            state_dim=state_dim, n_critics=n_critics, 
            critic_kwargs=critic_kwargs
        )
        v_critic_params = self.v_critics_def.init(v_critic_offline_key, dummy_obs)
        self.v_critics_state = TrainState.create(
            apply_fn=self.v_critics_def.apply,
            params=v_critic_params,
            tx=optax.adam(lr)
        )
        
        # Target Critics (Just params, no optimizer)
        self.v_target_critics_params = v_critic_params

        # 3. Alpha (Temperature)
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy
            
        self.alpha_def = AlphaModule(ent_init=self.alpha_init)
        alpha_params = self.alpha_def.init(alpha_offline_key)
        self.alpha_state = TrainState.create(
            apply_fn=self.alpha_def.apply,
            params=alpha_params,
            tx=optax.adam(lr)
        )

        # Replay Buffers
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, int(batch_size*self.gradient_steps), gamma, n_envs)
        self.replay_buffer_demo = ReplayBuffer(state_dim, action_dim, buffer_size, int(batch_size*self.gradient_steps), gamma, 1)
        # Hparams for logging
        self.hparams = {
            "state_dim": self.state_dim, "action_dim": self.action_dim, "tau": self.tau,
            "gamma": self.gamma, "lr": self.lr, "batch_size": self.batch_size,
            "n_critics": n_critics, "m_critics": m_critics, "seed": seed,
            "beta" : beta, "omega" : omega, "kappa" : kappa
        }

        # Logger
        if logger_name == "mlflow":
            self.logger = MLFlowLogger(uri=mlflow_uri, experiment_name=experiment_name, run_name=run_name)
            self.logger.start()
            self.logger.log_params(self.hparams)

    @staticmethod
    @partial(jax.jit, static_argnames="apply_fn")
    def _sample_action(apply_fn, params, obs, key):
        actions = apply_fn(params, obs, key, method=TanhGaussianPolicy.sample_without_probs)
        return actions
    
    @staticmethod
    @partial(jax.jit, static_argnames="apply_fn")
    def _sample_action_det(apply_fn, params, obs):
        actions = apply_fn(params, obs, method=TanhGaussianPolicy.sample_det)
        return actions
    
    def select_action_offline(self, state, deterministic=True):
        state = jnp.array(state)
        if state.ndim == 1: state = state[None, ...]
        
        if deterministic:
            action = ACA_Agent._sample_action_det(self.actor_offline_state.apply_fn, self.actor_offline_state.params, state)
        else:
            self.rng, key = jax.random.split(self.rng)
            action = ACA_Agent._sample_action(self.actor_offline_state.apply_fn, self.actor_offline_state.params, state, key)
            
        return np.array(action)
    
    def select_action_online(self, state, deterministic=True):
        state = jnp.array(state)
        if state.ndim == 1: state = state[None, ...]
        
        if deterministic:
            action = ACA_Agent._sample_action_det(self.actor_online_state.apply_fn, self.actor_online_state.params, state)
        else:
            self.rng, key = jax.random.split(self.rng)
            action = ACA_Agent._sample_action(self.actor_online_state.apply_fn, self.actor_online_state.params, state, key)
            
        return np.array(action)
    
    def load_demo_trajectories(self, demo_file, demo_env, nds=40, nds_name="CGL", n_load=-1, n_jobs=-1):
        load_demo_trajectories_parallel([self.replay_buffer_demo], demo_file, demo_env, nds, nds_name, self.gamma, n_load=n_load, n_jobs=n_jobs)
        return self.replay_buffer_demo
    
    # Batch Update Offline
    @staticmethod
    @partial(jax.jit, static_argnames=("gradient_steps", "policy_delay_update"))
    def batch_update_offline(
        actor_offline_state, critics_state, target_critics_params, alpha_state,
        v_critic_state, v_target_critics_params,
        batch_demo, omega,
        rng, gamma, tau, target_entropy,
        gradient_steps, policy_delay_update
    ):
        """
        SAC+ML offline-only training loop. JIT-safe via lax.scan + lax.cond.
        batch_demo: shape (gradient_steps * batch_size, ...)
        """

        def reshape_for_scan(x):
            return x.reshape((gradient_steps, -1, *x.shape[1:]))

        demo_sharded = jax.tree_util.tree_map(reshape_for_scan, batch_demo)

        init_carry = (
            actor_offline_state,
            critics_state,
            target_critics_params,
            alpha_state,
            v_critic_state,
            v_target_critics_params,
            rng,
            jnp.zeros(()),  # c_loss
            jnp.zeros(()),  # v_c_loss
            jnp.zeros(()),  # a_loss
            jnp.zeros(()),  # al_loss
            jnp.zeros(()),  # curr_alpha
        )

        def scan_step(carry, i):
            (
                actor_state, critics_state, target_critics_params, alpha_state,
                v_critic_state, v_target_critics_params,
                rng, c_loss, v_c_loss, a_loss, al_loss, curr_alpha
            ) = carry

            # --- slice micro-batch ---
            step_batch = jax.tree_util.tree_map(lambda x: x[i], demo_sharded)

            # --- critic + v_critic update ---
            rng, r_critic = jax.random.split(rng)
            critics_state, c_loss, v_critic_state, v_c_loss = _update_critic_sacml(
                critic_state=critics_state,
                target_critic_params=target_critics_params,
                actor_state=actor_state,
                alpha_state=alpha_state,
                v_critic_state=v_critic_state,
                v_target_critic_params=v_target_critics_params,
                batch=step_batch,
                rng=r_critic,
                gamma=gamma,
            )

            # --- soft target updates ---
            target_critics_params = optax.incremental_update(
                critics_state.params, target_critics_params, tau
            )
            v_target_critics_params = optax.incremental_update(
                v_critic_state.params, v_target_critics_params, tau
            )

            # --- actor + alpha update every policy_delay_update steps ---
            def do_actor_update(_):
                rng_inner, r_actor = jax.random.split(rng)
                new_actor, new_alpha, new_a_loss, new_al_loss, new_curr_alpha = (
                    _update_actor_and_alpha_sacml(
                        actor_state=actor_state,
                        critics_state=critics_state,
                        alpha_state=alpha_state,
                        batch=step_batch,
                        rng=r_actor,
                        target_entropy=target_entropy,
                        omega=omega,
                    )
                )
                return new_actor, new_alpha, new_a_loss, new_al_loss, jnp.squeeze(new_curr_alpha), rng_inner

            def skip_actor_update(_):
                return actor_state, alpha_state, a_loss, al_loss, curr_alpha, rng

            actor_state, alpha_state, a_loss, al_loss, curr_alpha, rng = jax.lax.cond(
                ((i + 1) % policy_delay_update) == 0,
                do_actor_update,
                skip_actor_update,
                operand=None,
            )

            new_carry = (
                actor_state, critics_state, target_critics_params, alpha_state,
                v_critic_state, v_target_critics_params,
                rng, c_loss, v_c_loss, a_loss, al_loss, curr_alpha
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            scan_step,
            init_carry,
            jnp.arange(gradient_steps),
        )

        (
            actor_state, critics_state, target_critics_params, alpha_state,
            v_critic_state, v_target_critics_params,
            _, c_loss, v_c_loss, a_loss, al_loss, curr_alpha
        ) = final_carry

        metrics = {
            "critic_loss":   c_loss,
            "v_critic_loss": v_c_loss,
            "actor_loss":    a_loss,
            "alpha_loss":    al_loss,
            "alpha":         curr_alpha,
        }

        return (
            actor_state, critics_state, target_critics_params, alpha_state,
            v_critic_state, v_target_critics_params, metrics
        )
    
    # Batch Update Offline
    @staticmethod
    @partial(jax.jit, static_argnames=("gradient_steps", "policy_delay_update"))
    def batch_update_online(
        actor_online_state, critics_state, target_critics_params, alpha_state,
        actor_offline_state, batch_online, beta, kappa,
        rng, gamma, tau, target_entropy,
        gradient_steps, policy_delay_update
    ):
        """
        SAC+ML offline-only training loop. JIT-safe via lax.scan + lax.cond.
        batch_demo: shape (gradient_steps * batch_size, ...)
        """

        def reshape_for_scan(x):
            return x.reshape((gradient_steps, -1, *x.shape[1:]))

        online_sharded = jax.tree_util.tree_map(reshape_for_scan, batch_online)

        init_carry = (
            actor_online_state,
            actor_offline_state,
            critics_state,
            target_critics_params,
            alpha_state,
            rng,
            jnp.zeros(()),  # c_loss
            jnp.zeros(()),  # a_loss
            jnp.zeros(()),  # al_loss
            jnp.zeros(()),  # curr_alpha
        )

        def scan_step(carry, i):
            (
                actor_state, actor_offline_state, critics_state, target_critics_params, 
                alpha_state, rng, c_loss, a_loss, al_loss, curr_alpha
            ) = carry

            # --- slice micro-batch ---
            step_batch = jax.tree_util.tree_map(lambda x: x[i], online_sharded)

            # --- critic + v_critic update ---
            rng, r_critic = jax.random.split(rng)
            critics_state, c_loss = _update_critic_aca_sac(
                critic_state=critics_state,
                target_critic_params=target_critics_params,
                actor_state=actor_state,
                alpha_state=alpha_state,
                batch=step_batch,
                rng=r_critic,
                gamma=gamma,
                actor_offline_state=actor_offline_state,
                kappa=kappa,
                beta=beta
            )

            # --- soft target updates ---
            target_critics_params = optax.incremental_update(
                critics_state.params, target_critics_params, tau
            )

            # --- actor + alpha update every policy_delay_update steps ---
            def do_actor_update(_):
                rng_inner, r_actor = jax.random.split(rng)
                new_actor, new_alpha, new_a_loss, new_al_loss, new_curr_alpha = (
                    _update_actor_and_alpha_aca_sac(
                        actor_state=actor_state,
                        critics_state=critics_state,
                        alpha_state=alpha_state,
                        batch=step_batch,
                        rng=r_actor,
                        target_entropy=target_entropy,
                        actor_offline_state=actor_offline_state,
                        kappa=kappa,
                        beta=beta
                    )
                )
                return new_actor, new_alpha, new_a_loss, new_al_loss, new_curr_alpha, rng_inner

            def skip_actor_update(_):
                return actor_state, alpha_state, a_loss, al_loss, curr_alpha, rng

            actor_state, alpha_state, a_loss, al_loss, curr_alpha, rng = jax.lax.cond(
                ((i + 1) % policy_delay_update) == 0,
                do_actor_update,
                skip_actor_update,
                operand=None,
            )

            new_carry = (
                actor_state, actor_offline_state, critics_state, target_critics_params, 
                alpha_state, rng, c_loss, a_loss, al_loss, curr_alpha
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            scan_step,
            init_carry,
            jnp.arange(gradient_steps),
        )

        (
            actor_state, actor_offline_state, critics_state, target_critics_params, 
                alpha_state, rng, c_loss, a_loss, al_loss, curr_alpha
        ) = final_carry

        metrics = {
            "critic_loss":   c_loss,
            "actor_loss":    a_loss,
            "alpha_loss":    al_loss,
            "alpha":         curr_alpha,
        }

        return (
            actor_state, critics_state, target_critics_params, alpha_state, metrics
        )
    
    def actor_critic_allignment(self):
        # Transfering the actor
        self.actor_online_state = self.actor_online_state.replace(params=self.actor_offline_state.params)
        # Transfering the critic
        self.rng, critic_key = jax.random.split(self.rng)
        new_critics_params = init_critic_from_baseline(
            baseline_params=self.v_critics_state.params,
            critic_params=self.critics_state.params,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            noise_scale=1e-4,
            rng=critic_key
        )
        new_critic_opt = self.critics_state.tx.init(new_critics_params)
        self.critics_state = self.critics_state.replace(
            params=new_critics_params,
            opt_state=new_critic_opt,
            step=0
        )

        self.rng, critic_key2 = jax.random.split(self.rng)
        new_target_critic_params = init_critic_from_baseline(
            baseline_params=self.v_target_critics_params,
            critic_params=self.target_critics_params,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            noise_scale=1e-4,
            rng=critic_key2
        )
        self.target_critics_params = new_target_critic_params
        # Transfering the alpha
        self.rng, alpha_key = jax.random.split(self.rng)
        self.alpha_def = AlphaModule(target_entropy=self.target_entropy)
        alpha_params = self.alpha_def.init(alpha_key)
        self.alpha_state = TrainState.create(
            apply_fn=self.alpha_def.apply,
            params=alpha_params,
            tx=optax.adam(self.lr)
        )


    def online_train(
            self, 
            env,
            total_online_training_steps=1_000_000,
            learning_starts=5_000,
            progress_bar = True,
            verbose = 1,
            log_interval = 5,
            log_interval_metric = 5_000,
            callback = None
    ):
        self.total_online_training_steps = total_online_training_steps
        self._count_total_gradients_taken = 0
        if callback: callback.on_training_start(self)
        pbar = tqdm(total=total_online_training_steps) if progress_bar else None

        obs, _ = env.reset()
        _episode_start = np.zeros(env.num_envs, dtype=bool)
        _episode_rewards = np.zeros(env.num_envs)
        _episode_lengths = np.zeros(env.num_envs)
        
        self._total_timesteps_ran = 0
        self.logger_count = 1

        while self._total_timesteps_ran <= self.total_online_training_steps:
            # Action Selection
            self.rng, key = jax.random.split(self.rng)
            actions = self.select_action(obs, deterministic=False)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = np.logical_or(terminations, truncations)

            for i in range(env.num_envs):
                if not _episode_start[i]:
                    self.replay_buffer.add(obs[i], actions[i], rewards[i], terminations[i], truncations[i], next_obs[i])
                else:
                    self.logger_count += 1
                    self._ep_lengths.append(_episode_lengths[i])
                    self._ep_rewards.append(_episode_rewards[i])
                    _episode_rewards[i], _episode_lengths[i] = 0, -1

            _episode_rewards += rewards
            _episode_lengths += 1
            self._total_timesteps_ran += env.num_envs
            obs = next_obs
            _episode_start = dones

            # Update Step
            if self._total_timesteps_ran >= learning_starts and self._total_timesteps_ran % self.train_freq == 0:
                    # Sample Buffers
                    batch = self.replay_buffer.sample()

                    # Update
                    self.rng, update_key = jax.random.split(self.rng)
                             
                    (self.actor_online_state, self.critics_state, self.target_critics_params, 
                    self.alpha_state, metrics) = ACA_Agent.batch_update_online(
                        self.actor_offline_state,
                        self.critics_state,
                        self.target_critics_params,
                        self.alpha_state,
                        self.actor_offline_state,
                        batch,
                        self.beta,
                        self.kappa,
                        update_key,
                        self.gamma,
                        self.tau,
                        self.target_entropy,
                        self.gradient_steps,
                        self.policy_delay_update
                    )
                    self._count_total_gradients_taken += self.gradient_steps        

            # Logging
            if self._total_timesteps_ran >= learning_starts and self.logger_count % log_interval == 0:
                mean_rew = np.mean(self._ep_rewards) if self._ep_rewards else 0
                mean_len = np.mean(self._ep_lengths) if self._ep_lengths else 0
                fps = self._total_timesteps_ran / (time() - self._start_time)

                self.logger.log_metric("rollout/mean_episode_length", mean_len, step=self._total_timesteps_ran)
                self.logger.log_metric("rollout/mean_episode_reward", mean_rew, step=self._total_timesteps_ran)
                self.logger.log_metric("rollout/frames_per_second", fps, step=self._total_timesteps_ran)
                
                if verbose:
                    tqdm.write("-"*50)
                    tqdm.write(f" Step: {self._total_timesteps_ran:<8d}")
                    tqdm.write(f" MeanEpLen: {mean_len:.2f}")
                    tqdm.write(f" MeanEpRew: {mean_rew:.2f}")
                    tqdm.write("-"*50)
                
                self.logger_count = 1

            if self._total_timesteps_ran >= learning_starts and self._count_total_gradients_taken % log_interval_metric == 0:
                self.logger.log_metric("training/actor_loss", metrics['actor_loss'].item(), step=self._total_timesteps_ran)
                self.logger.log_metric("training/critic_loss", metrics['critic_loss'].item(), step=self._total_timesteps_ran)
                self.logger.log_metric("training/alpha_loss", metrics['alpha_loss'].item(), step=self._total_timesteps_ran)
                self.logger.log_metric("training/alpha", metrics['alpha'].item(), step=self._total_timesteps_ran)

                if verbose:
                    tqdm.write("-"*50)
                    tqdm.write(f" Actor_Loss: {metrics['actor_loss'].item():<2f}")
                    tqdm.write(f" Critic_Loss: {metrics['critic_loss'].item():.2f}")
                    tqdm.write(f" Alpha_Loss: {metrics['alpha_loss'].item():.2f}")
                    tqdm.write(f" Alpha: {metrics['alpha'].item():.2f}")
                    tqdm.write("-"*50)

            if callback: callback.on_step(self._total_timesteps_ran,self)
            if pbar: pbar.update(env.num_envs)

        if callback: callback.on_training_end(self)
        return self.actor_online_state 

    def offline_train(
            self, 
            total_offline_training_steps=5_00_000,
            progress_bar = True,
            verbose = 1,
            log_interval = 5,
            log_interval_metric = 5_000,
            callback = None
    ):
        self.total_offline_training_steps = total_offline_training_steps
        self._count_total_gradients_taken = 0
        if callback: callback.on_training_start(self)
        pbar = tqdm(total=total_offline_training_steps) if progress_bar else None

        while self._count_total_gradients_taken <= self.total_offline_training_steps:
            batch_demo = self.replay_buffer_demo.sample()
            # Update key
            self.rng, update_key = jax.random.split(self.rng)
            (self.actor_offline_state, self.critics_state, self.target_critics_params, 
             self.alpha_state, self.v_critics_state, self.v_target_critics_params, metrics) = ACA_Agent.batch_update_offline(
                 self.actor_offline_state,
                 self.critics_state,
                 self.target_critics_params,
                 self.alpha_state,
                 self.v_critics_state,
                 self.v_target_critics_params,
                 batch_demo,
                 self.omega,
                 update_key,
                 self.gamma,
                 self.tau,
                 self.target_entropy,
                 self.gradient_steps,
                 self.policy_delay_update
             )
            
            self._count_total_gradients_taken += self.gradient_steps
            if  pbar : pbar.update(self.gradient_steps)
            if callback: callback.on_step(self._count_total_gradients_taken,self)

            if (self._count_total_gradients_taken) % log_interval_metric == 0:
                self.logger.log_metric("offline/critic_loss", metrics["critic_loss"], step=self._count_total_gradients_taken)
                self.logger.log_metric("offline/actor_loss", metrics["actor_loss"], step=self._count_total_gradients_taken)
                self.logger.log_metric("offline/v_critic_loss", metrics["v_critic_loss"], step=self._count_total_gradients_taken)
                self.logger.log_metric("offline/alpha_loss", metrics["alpha_loss"], step=self._count_total_gradients_taken)
                self.logger.log_metric("offline/alpha", metrics["alpha"], step=self._count_total_gradients_taken)

                if verbose:
                    tqdm.write("-"*50)
                    tqdm.write(f"Step: {self._count_total_gradients_taken:<8d}")
                    tqdm.write(f" Critic Loss: {metrics['critic_loss']:.2f}")
                    tqdm.write(f" Actor Loss: {metrics['actor_loss']:.2f}")
                    tqdm.write(f" Value Critic Loss: {metrics['v_critic_loss']:.2f}")
                    tqdm.write(f" Alpha Loss: {metrics['alpha_loss']:.2f}")
                    tqdm.write(f" Alpha: {metrics['alpha']:.2f}")
                    tqdm.write("-"*50)

        if callback: callback.on_training_end(self)
        if pbar: pbar.close()  