# Fixed Trajectory Impedance Control

This example trains the Franka drawer-opening task using a fixed Cartesian trajectory.
Only the stiffness and damping are learned by the policy.

Run training with:

```bash
./isaaclab.sh -p train_fixed_traj_impedance.py --num_envs 32 --headless
```
