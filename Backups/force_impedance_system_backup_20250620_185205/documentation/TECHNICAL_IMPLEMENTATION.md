# Technical Implementation Details

## 🔧 Core Components

### 1. Force Variable Impedance Environment Configuration
**File**: `force_variable_impedance_env_cfg.py`

**Key Features**:
- Inherits from base `CabinetEnvCfg`
- Configures 19D action space with operational space control
- Sets up contact sensors on `panda_hand`
- Implements curriculum learning with 3 phases
- Defines 11 reward terms for comprehensive learning

**Critical Configuration**:
```python
# Operational Space Control with Variable Impedance
self.actions.arm_action = OperationalSpaceControllerActionCfg(
    impedance_mode="variable",  # Enables 6D stiffness + 6D damping learning
    controller=OperationalSpaceControllerCfg(
        target_type="pose_rel",  # 6D relative pose control
        stiffness={'.*': 600.0},  # Initial stiffness (learned)
        damping_ratio={'.*': 1.0}  # Initial damping (learned)
    )
)

# Contact Force Sensing
self.scene.robot.spawn.activate_contact_sensors = True
```

### 2. MDP Functions for Force and Impedance
**File**: `force_impedance_mdp.py`

**Custom Observations**:
- `contact_force_magnitude`: Real-time force feedback
- `current_impedance_params`: Current stiffness/damping state
- Integrated into observation manager for policy input

### 3. PPO Training Configuration  
**File**: `rsl_rl_ppo_cfg.py`

**Optimized Hyperparameters**:
- Learning rate: 5e-4 (balanced exploration/exploitation)
- Network: [512, 256, 128] (sufficient capacity for 19D actions)
- Entropy coefficient: 0.006 (encourages exploration)
- Max iterations: 1200 (curriculum progression)

### 4. Training Pipeline
**File**: `train_force_variable_impedance_direct.py`

**Key Features**:
- Direct configuration instantiation (bypasses gym registration issues)
- RSL-RL integration with proper environment wrapping
- Automatic logging and checkpoint management
- Configuration backup for reproducibility

## 🧠 Neural Network Architecture

### Actor Network (Policy)
```
Input: 55D observations
  ↓
Layer 1: 55 → 512 (ReLU)
  ↓  
Layer 2: 512 → 256 (ReLU)
  ↓
Layer 3: 256 → 128 (ReLU) 
  ↓
Output: 128 → 19 (Tanh) [Actions]
```

### Critic Network (Value Function)
```
Input: 55D observations
  ↓
Layer 1: 55 → 512 (ReLU)
  ↓
Layer 2: 512 → 256 (ReLU)
  ↓  
Layer 3: 256 → 128 (ReLU)
  ↓
Output: 128 → 1 (Linear) [Value]
```

## 📊 Action Space Breakdown

**Total: 19 Dimensions**

### Arm Control (18D):
1. **Pose Control (6D)**: [x, y, z, roll, pitch, yaw] relative commands
2. **Stiffness Parameters (6D)**: [Kx, Ky, Kz, Krx, Kry, Krz] 
3. **Damping Parameters (6D)**: [Dx, Dy, Dz, Drx, Dry, Drz]

### Gripper Control (1D):
1. **Binary Position**: Open/close command

## 🎯 Observation Space Breakdown

**Total: 55 Dimensions**

### Robot State (18D):
- Joint positions (9D): Franka arm + gripper
- Joint velocities (9D): Franka arm + gripper

### Task Context (6D):
- Cabinet joint position (1D): Drawer state
- Cabinet joint velocity (1D): Drawer dynamics  
- End-effector to drawer distance (3D): Spatial relationship
- [Additional context as needed]

### Force Feedback (1D):
- Contact force magnitude: Real-time force sensing

### Impedance State (12D):
- Current stiffness parameters (6D): Active compliance state
- Current damping parameters (6D): Active damping state

### Action History (19D):
- Previous actions: Enables temporal learning patterns

## 🏆 Reward System Design

### Manipulation Rewards (Positive):
1. **approach_ee_handle** (6.0): Encourage handle approach
2. **align_ee_handle** (0.5): Proper end-effector alignment  
3. **approach_gripper_handle** (15.0): Gripper positioning
4. **align_grasp_around_handle** (0.125): Grasp alignment
5. **grasp_handle** (8.0): Successful grasping
6. **open_drawer_bonus** (20.0): Task completion reward
7. **multi_stage_open_drawer** (5.0): Progressive opening

### Efficiency Penalties (Negative):
8. **action_rate_l2** (-0.01): Smooth action execution
9. **joint_vel** (-0.0001): Energy efficiency

### Force/Impedance Rewards (Curriculum):
10. **contact_force_penalty** (0.0 → active): Gentle manipulation
11. **impedance_adaptation** (0.0 → active): Optimal compliance

## 🔄 Curriculum Implementation

### Phase Transitions:
- **Triggers**: Episode count or performance thresholds
- **Reward Scaling**: Gradual introduction of force/impedance terms
- **Complexity**: Progressive increase in learning objectives

### Implementation:
```python
# Curriculum terms in environment configuration
self.curriculum.grasp_focus_curriculum = CurriculumTermCfg(
    func=mdp.modify_reward_weight,
    params={"term_name": "approach_gripper_handle", "weight": (5.0, 15.0), "num_steps": 300}
)
```

## 🚀 Training Process

### 1. Initialization:
- Load environment and agent configurations
- Create 19D action space neural networks
- Initialize curriculum learning parameters

### 2. Phase 1 Training (Episodes 0-300):
- Focus on basic manipulation skills
- Force and impedance rewards disabled
- Learn handle approach and grasping

### 3. Phase 2 Training (Episodes 300-600):
- Introduce contact force penalties
- Maintain fixed impedance parameters
- Learn force-aware manipulation

### 4. Phase 3 Training (Episodes 600-900):
- Enable impedance parameter adaptation
- Activate impedance optimization rewards
- Learn variable compliance control

### 5. Phase 4 Training (Episodes 900-1200):
- Full reward system active
- Master force-impedance coordination
- Optimize for task performance and safety

### 6. Evaluation:
- Test on various manipulation scenarios
- Analyze force profiles and impedance adaptation
- Compare with baseline joint position control
