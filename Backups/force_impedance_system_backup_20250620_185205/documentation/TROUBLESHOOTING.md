# Troubleshooting Guide

## 🔧 Common Issues and Solutions

### 1. Environment Creation Failures

**Issue**: `TypeError: ManagerBasedRLEnv.__init__() missing 1 required positional argument: 'cfg'`

**Solution**: Use direct configuration approach instead of gym.make():
```python
# ❌ Problematic
env = gym.make("Isaac-Open-Drawer-Franka-Force-Variable-Impedance-v0")

# ✅ Working solution  
env_cfg = ForceVariableImpedanceCabinetEnvCfg()
env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=env_cfg)
```

### 2. Action Space Dimension Mismatch

**Issue**: `ValueError: Invalid action shape, expected: 19, received: X`

**Diagnosis**: Check action space shape interpretation:
```python
# Action space shows (num_envs, action_dim)
action_space.shape  # Should be (num_envs, 19) not (19,)
action_dim = action_space.shape[1]  # Use index 1, not 0
```

**Solution**: Ensure test scripts check correct dimension index.

### 3. Contact Sensor Configuration

**Issue**: Contact sensors not working or collision errors

**Solution**: Verify contact sensor setup:
```python
# Enable contact sensors in robot spawn config
self.scene.robot.spawn.activate_contact_sensors = True

# Use correct body name for contact sensing
contact_sensor_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Use hand, not fingers
    filter_prim_paths_expr=[],
    history_length=1
)
```

### 4. Import and Registration Issues

**Issue**: Module import errors or environment not found

**Solution**: Check registration in `__init__.py`:
```python
# Ensure proper registration
gym.register(
    id="Isaac-Open-Drawer-Franka-Force-Variable-Impedance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "path.to.config:ForceVariableImpedanceCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": "path.to.agents:ForceVariableImpedanceCabinetPPORunnerCfg"
    }
)
```

### 5. Training Performance Issues

**Issue**: Slow training or poor learning progress

**Solutions**:
- **Check GPU utilization**: Ensure CUDA is properly configured
- **Adjust batch size**: Balance memory usage vs. training speed  
- **Verify reward scaling**: Check reward magnitudes are appropriate
- **Monitor curriculum**: Ensure progressive learning is working

### 6. Network Architecture Mismatches

**Issue**: Network input/output dimension errors

**Solution**: Verify dimensions match:
```python
# Check observation space
print(f"Obs space: {env.observation_space}")  
# Should show 55D observations

# Check action space  
print(f"Action space: {env.action_space}")
# Should show (num_envs, 19) 

# Verify network architecture
print(f"Actor output: {policy.action_dim}")  # Should be 19
print(f"Critic input: {policy.obs_dim}")     # Should be 55
```

## 🚨 Emergency Debugging

### Quick Validation Test
```python
# Run minimal validation
env_cfg = ForceVariableImpedanceCabinetEnvCfg()
env_cfg.scene.num_envs = 1  # Minimal for debugging
env = gym.make("Isaac-Open-Drawer-Franka-v0", cfg=env_cfg)

# Check dimensions
print(f"Action: {env.action_space.shape}")      # (1, 19)
print(f"Obs: {env.observation_space}")          # Should include 'policy' key

# Test reset and step
obs = env.reset()
action = torch.zeros((1, 19), device=env.device)  # Minimal action
obs, reward, done, info = env.step(action)
print("✅ Basic functionality working")
```

### Log Analysis
```bash
# Check training logs for issues
tail -f logs/rsl_rl/franka_force_impedance_curriculum/latest/progress.txt

# Look for:
# - Reward progression
# - Episode length changes  
# - Network loss values
# - Curriculum phase transitions
```

## 📊 Performance Validation

### Expected Training Metrics:
- **Initial Mean Reward**: 15-25
- **Learning Progression**: Steady increase over iterations
- **Episode Length**: Should increase as policy improves
- **Handle Approach Reward**: Major component of success

### Red Flags:
- **Flat Learning Curves**: Check reward scaling or network capacity
- **Exploding Values**: Reduce learning rate or clip gradients  
- **Early Termination**: Check termination conditions
- **NaN Values**: Numerical instability, check input ranges

## 🔄 Recovery Procedures

### If Training Fails:
1. **Backup current state**: Save logs and configurations
2. **Reduce complexity**: Start with simpler curriculum
3. **Check configurations**: Validate all parameters
4. **Test environment**: Run standalone environment tests
5. **Gradual debugging**: Add complexity incrementally

### If Environment Breaks:
1. **Restart Isaac Sim**: Clear simulation context
2. **Check device settings**: Ensure CUDA compatibility
3. **Validate configurations**: Test with minimal setups
4. **Review recent changes**: Identify breaking modifications
