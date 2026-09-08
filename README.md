# PyBullet PPO Navigation

A self-contained PyBullet navigation environment trained with Proximal Policy Optimization (PPO). The project recreates a corridor-style robot navigation task with lidar observations, static and dynamic obstacles, collision detection, checkpointing, evaluation, and TensorBoard logging.

## Features

- 36-beam lidar plus goal distance and heading observations
- PPO actor-critic training with GAE
- Static corridor obstacles and an optional moving obstacle
- Real PyBullet contact-based collision detection
- Automatic checkpoint resume
- CUDA, Apple Silicon MPS, and CPU device selection
- GUI, fast GUI, and headless training modes
- TensorBoard-compatible training logs

## Requirements

- Python 3.9 or newer
- PyTorch
- NumPy
- PyBullet

Install the Python dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy torch pybullet
```

For Apple Silicon, install the PyTorch build appropriate for your machine if the default package is not suitable.

## Run Training

From the repository root:

```bash
# Open the PyBullet GUI and run in real time
python train_agent_pybullet.py

# Open the GUI but run the simulation as fast as possible
python train_agent_pybullet.py --fast

# Run without a window; recommended for long training runs
python train_agent_pybullet.py --headless
```

The trainer automatically uses CUDA when available, then Apple Silicon MPS, and finally CPU.

## Configuration

Edit `pybullet_config.py` to change PPO, reward, environment, obstacle, evaluation, or simulation settings. The training entry point assigns the included `robot.urdf` automatically. If you use another robot, update `build_config()` in `train_agent_pybullet.py` and set the wheel joint names, wheel geometry, and lidar offsets as needed.

Useful environment settings include:

- `enable_obstacles`: enable or disable the corridor obstacles
- `dynamic_obstacle_enabled`: enable or disable the moving obstacle
- `target_goal`: goal position in world coordinates
- `max_steps_per_episode`: episode length limit
- `workspace_dir`: location for models, logs, TensorBoard data, and saved configs

## Output Files

By default, generated training data is stored in `~/pybullet_nav/MODEL/`:

```text
MODEL/
├── configs/       # Saved training configuration
├── logs/          # Episode and evaluation logs
├── models/        # PPO checkpoints
└── tensorboard/   # TensorBoard event files
```

These generated directories are intentionally not committed to the repository.

## TensorBoard

Launch TensorBoard while training or after a run:

```bash
tensorboard --logdir ~/pybullet_nav/MODEL/tensorboard
```

## Project Structure

- `train_agent_pybullet.py` - PPO training and evaluation entry point
- `pybullet_nav_env.py` - PyBullet world, observations, actions, rewards, and resets
- `pybullet_config.py` - Environment and PPO configuration
- `ppo_model.py` - Actor-critic neural network
- `checkpoint_manager.py` - Checkpoint saving and resume support
- `logger.py` - Running normalization and training logs
- `robot.urdf` - Example four-wheel robot model

## Notes

The included environment is synchronous: each call to `reset()` fully resets the robot, velocities, obstacle timer, and waypoint state. Training can resume from the latest compatible checkpoint in the configured models directory.
