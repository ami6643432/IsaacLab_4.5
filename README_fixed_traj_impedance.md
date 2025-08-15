# Fixed Trajectory Impedance Control

This example trains the Franka drawer-opening task using a fixed Cartesian trajectory.
Only the stiffness and damping parameters are learned by the policy, while the end-effector
follows a predefined path.

## Overview

The implementation uses operational space control with the following design:

- **Fixed Trajectory**: A predefined sequence of waypoints in Cartesian space
- **Variable Impedance**: Policy learns optimal stiffness and damping parameters
- **Force Feedback**: Contact forces are used as observations for the policy
- **Curriculum Learning**: Progressive training focuses on manipulation and impedance

## Training

Run training with:

```bash
./isaaclab.sh -p train_fixed_traj_impedance.py --num_envs 32 --headless
```

Additional options:
- `--max_iterations 2000`: Set the number of training iterations
- `--checkpoint PATH`: Resume training from a checkpoint
- `--seed VALUE`: Set random seed for reproducibility

## Implementation Details

The system implements a 12-dimensional action space:
- 6D stiffness parameters [Kx, Ky, Kz, Krx, Kry, Krz]
- 6D damping parameters [Dx, Dy, Dz, Drx, Dry, Drz]

The fixed trajectory consists of 4 key waypoints:
1. Approach handle
2. At handle
3. Pull drawer
4. Fully open

## Results and Evaluation

The trained policy should demonstrate:
- Appropriate impedance adaptation based on manipulation phase
- Reduced contact forces during interaction
- Smooth and successful drawer opening
