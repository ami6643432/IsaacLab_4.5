# How to Run Variable Impedance Environment

## Important Note on Running Scripts

The key issue identified is that all scripts must be run using the IsaacLab launcher script (`isaaclab.sh` or `isaaclab.bat`) to access the Omniverse Kit libraries like `omni.log`. Regular Python interpreters don't have access to these libraries.

## Running the Environment

To properly run and test your variable impedance environment:

1. **Test basic imports** (to verify setup):
   ```bash
   cd /home/amitabh/IsaacLab
   ./isaaclab.sh -p data/singlearm/test_basic_imports.py
   ```

2. **Quick test** (verifies environment can be created):
   ```bash
   cd /home/amitabh/IsaacLab
   ./isaaclab.sh -p data/singlearm/quick_test.py
   ```

3. **Run the training test script** (tests actions and rewards):
   ```bash
   cd /home/amitabh/IsaacLab
   ./isaaclab.sh -p data/singlearm/train_variable_impedance_cabinet.py --num_envs 4
   ```

4. **Run actual training with RL framework**:
   ```bash
   cd /home/amitabh/IsaacLab
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Variable-Impedance-Cabinet-v0 --num_envs 512
   ```

5. **Run in play mode** (with a trained policy):
   ```bash
   cd /home/amitabh/IsaacLab
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Variable-Impedance-Cabinet-Play-v0
   ```

## Common Issues

1. **ModuleNotFoundError: No module named 'omni.log'** - This indicates you're not using the IsaacLab launcher. Always run your scripts using `./isaaclab.sh -p your_script.py`.

2. **Import errors for custom modules** - The environment registration in `__init__.py` is important for making your environment discoverable by the gym registry.

3. **Path issues** - Don't manually add paths with `sys.path.append()` when using IsaacLab. The launcher sets up the paths correctly.

4. **Performance issues** - For slower machines, reduce `num_envs` to a smaller number (e.g., 4, 8, or 16) for testing.

## Additional Debugging Tips

If you need to check specific Python paths or environment details:
```bash
./isaaclab.sh -p -c "import sys; print(sys.path); import omni.log; print('omni.log imported successfully')"
```

To check all registered environments:
```bash
./isaaclab.sh -p -c "import gymnasium as gym; print([env for env in gym.registry if 'Isaac' in env])"
```
