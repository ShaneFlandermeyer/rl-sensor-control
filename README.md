# rl-sensor-control

## Usage

To run experiments, you must install this repository and its dependencies as follows
```[bash]
pip install -e .
```

This installs the following packages:
- [MOTPY](https://github.com/ShaneFlandermeyer/MOTpy/tree/develop): Multi-object tracking in python
- [flax-gnn](https://github.com/ShaneFlandermeyer/flax-gnn/tree/develop): Graph neural networks in Jax/Flax
- [tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax/tree/develop): Model-based RL engine
- [BMPC](https://github.com/ShaneFlandermeyer/bmpc-jax/tree/develop): Extension of TD-MPC2. Not currently used, as I expect it to only improve performance in large action spaces.