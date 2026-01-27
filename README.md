# RL_JAX_AGENTS
Own Implementations of RL algorithms for continuous domains (action and state space) which would work for gym compatible environments with pytorch interface. This repository
mainly focusses on my problem statement. Uses a MLP Policy and is predominantly focussed on learning from demonstrations.

## Implemented Algorithms
- RLPD (Reinforcement Learning from Prior Data) \
Link : https://github.com/ikostrikov/rlpd \
The implementation here is derived from the above repository from the author with minor changes and more simplicity using Jax.

- Actor Critic Allignment for Offline-to-Online RL \
Link : https://github.com/ZishunYu/Actor-Critic-Alignment \
The implementation is in PyTorch and for d4rl environment. I would be using Jax and Gym compatible Environments