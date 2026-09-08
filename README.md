# MOANS

## Mapless autonomous navigation for indoor robots

<div align="center">

![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-22314E?style=for-the-badge&logo=ros)
![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-not%20specified-lightgrey?style=for-the-badge)

**A ROS 2 workspace for experimenting with perception, mapping, control, and learning-based navigation in dynamic indoor environments.**

</div>

---

## Why MOANS?

Most indoor navigation demos assume that a finished map already exists. MOANS is structured around the harder, more useful problem: a robot must build understanding while it moves through an unfamiliar space.

The repository brings together:

- ROS 2 launch files for simulation, visualization, robot spawning, and Cartographer.
- A Python navigation package with configuration, logging, checkpoints, PPO, and training utilities.
- Standalone C++ solutions for three algorithmic practice problems.
- A small `control.py` utility for coordinating the local workflow.

## Architecture at a glance

```text
Sensors / simulation
        |
        v
  Perception and mapping  --->  Robot pose
        |                         |
        +------> Navigation <-----+
                    |
                    v
             Velocity commands

Learning loop: observations -> PPO policy -> action -> environment -> reward
```

The project is intentionally modular: simulation and ROS 2 launch concerns live in `src/my_project/launch`, while navigation and learning code live in `src/my_project/my_project`.

## Repository map

```text
.
├── src/my_project/
│   ├── launch/              # Gazebo, RViz, Cartographer, and spawn launch files
│   ├── my_project/          # Navigation, PPO, logging, and checkpoint modules
│   ├── test/                # Package quality checks
│   ├── package.xml
│   └── setup.py
├── control.py               # Local workflow helper
├── A_Games_on_the_Train.cpp
├── B_Tatar_TV_Show.cpp
├── C_Omsk_Programmers.cpp
└── README.md
```

Generated ROS 2 directories (`src/build`, `src/install`, and `src/log`) are included in the upstream snapshot for reproducibility. For new development, rebuild them locally instead of committing fresh generated output.

## Prerequisites

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.10 or newer
- `colcon`
- Gazebo and RViz 2, when running the simulation launch files

Source ROS 2 before building:

```bash
source /opt/ros/jazzy/setup.bash
```

## Build

```bash
cd MOANS
colcon build --symlink-install --base-paths src/my_project
source install/setup.bash
```

If you are working from the repository as a ROS workspace, the package can also be built with the normal workspace command:

```bash
colcon build --symlink-install
source install/setup.bash
```

## Run

Launch the available workflows through the package:

```bash
ros2 launch my_project gazebo.launch.py
ros2 launch my_project spawn_robot.launch.py
ros2 launch my_project display.launch.py
ros2 launch my_project cartographer.launch.py
```

The learning entry point is available after the package is built:

```bash
ros2 run my_project train_agent
```

For a quick look at the Python modules without launching ROS:

```bash
python3 -m py_compile src/my_project/my_project/*.py
```

## Development notes

The package contains the following core pieces:

| Area | Modules |
| --- | --- |
| Training | `train_agent.py`, `ppo_model.py` |
| Navigation | `drl_navigator.py`, `config.py` |
| Experiment support | `checkpoint_manager.py`, `logger.py` |
| ROS 2 integration | `launch/*.launch.py`, `setup.py`, `package.xml` |

The C++ files at the repository root are independent console programs. Compile one with:

```bash
clang++ -std=c++17 -O2 A_Games_on_the_Train.cpp -o games_on_the_train
./games_on_the_train
```

Replace the source and output names to run either of the other two solutions.

## Current scope

MOANS is an active research and learning workspace rather than a packaged robot product. Hardware-specific drivers, benchmark results, and a project license are not defined in the current source tree. Contributions that make experiments repeatable, improve simulation fidelity, or add measurable navigation evaluations are especially useful.

## Author

**Devendra Kumar**

Repository: [github.com/Dk8494/MOANS](https://github.com/Dk8494/MOANS)
