# Accelerated Methods for Deep Reinforcement Learning

Code behind "Accelerated Methods for Deep Reinforcement Learning" by Adam Stooke and Pieter Abbeel: [https://arxiv.org/abs/1803.02811](https://arxiv.org/abs/1803.02811).  Includes single- and multi-GPU versions (both synchronous and asynchronous learners) of A2C, PPO, DQN + variants--all optimized for speed--along with an optimized implementation of the Atari environment (using ALE).

Borrows from the original rllab (including installation instructions), with documentation at: [https://rllab.readthedocs.org/en/latest/](https://rllab.readthedocs.org/en/latest/).

Examples are in accel_rl/scripts/examples, and include bare "train" scripts, which can be called directly, and associated "run" scripts, which invoke multiple train scripts to run experiment sweeps.

