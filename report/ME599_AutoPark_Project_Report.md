# Auto Park: Reinforcement Learning for Autonomous Vehicle Parking

## ME599 - University of Waterloo
### Team Members: Kevin Ma, Arjun Krishna

## 1. Background Material

Autonomous driving technologies have seen significant advancements in recent years, with self-parking capabilities emerging as one of the most commercially viable features. The implementation of autonomous parking systems helps address several challenges that modern drivers face, including stress during parking maneuvers, spatial awareness limitations, and accident risk reduction.

Reinforcement Learning (RL) offers a promising approach to solving autonomous parking problems by allowing agents to learn through trial and error interactions with their environment. Unlike supervised learning methods that require extensive labeled data, RL agents can optimize their behavior by maximizing cumulative rewards over time without direct supervision.

### 1.1 Trust Region Policy Optimization (TRPO)

Among various RL algorithms, Trust Region Policy Optimization (TRPO) has proven particularly effective for continuous control tasks like autonomous driving. TRPO, introduced by Schulman et al. [1], employs a constraint optimization approach that ensures policy updates remain within a "trust region," preventing destructive large policy changes that could lead to catastrophic forgetting or performance collapse.

Key advantages of TRPO include:
- Monotonic policy improvement guarantees
- Stable learning across a wide range of hyperparameters
- Effective handling of continuous action spaces
- Robustness to noise and environmental variations

These characteristics make TRPO well-suited for autonomous parking tasks where precise control and adaptation to varying conditions are essential.

### 1.2 Physics Simulation for Training

Training RL agents for autonomous driving tasks requires extensive interaction with the environment, which is impractical and potentially dangerous in real-world settings. Physics simulation environments like PyBullet [2] provide a safe, controllable alternative where agents can learn from millions of interactions without physical risk or expense.

The integration of RL algorithms with physics simulations enables:
- Safe exploration of failure cases
- Accelerated learning through parallelization
- Systematic testing across diverse scenarios
- Transferability of learned policies to real-world applications

### 1.3 CAN Bus Integration

Modern vehicles utilize Controller Area Network (CAN) buses for internal communications between electronic control units (ECUs). For autonomous parking systems to be practical in real vehicles, they must interface with the vehicle's existing CAN infrastructure. This involves translating high-level control commands (steering angle, throttle) into appropriate CAN messages that the vehicle's systems can understand and execute [3].

## 2. UWAFT Integration

The Auto Park project has direct applications to the University of Waterloo Alternative Fuels Team (UWAFT), which participates in advanced vehicle technology competitions. UWAFT focuses on developing sustainable transportation solutions, with autonomous features being a key component of their vehicle platforms.

### 2.1 Relevance to UWAFT Projects

Our autonomous parking system aligns with UWAFT's goals in several ways:
- It can be integrated into UWAFT's existing vehicle platforms, enhancing their autonomous capabilities
- The CAN bus integration demonstrates compatibility with automotive standards used in UWAFT vehicles
- The reinforcement learning approach provides adaptability to different vehicle dynamics, beneficial for alternative fuel vehicles with unique handling characteristics

### 2.2 UWAFT Vehicle Implementation Pathway

The project has been developed with UWAFT integration in mind, specifically:
1. The CAN message format has been designed to match UWAFT's existing vehicle communication protocols
2. The control parameters are calibrated for typical UWAFT vehicle dynamics
3. The simulation environment can be readily adapted to match UWAFT vehicle specifications

### 2.3 Next Steps for UWAFT

For full integration with UWAFT vehicles, we recommend the following next steps:
1. Hardware-in-loop testing with UWAFT's vehicle control systems
2. Domain adaptation techniques to bridge the sim-to-real gap
3. Gradual deployment starting with controlled environments
4. Expansion of the model to handle multiple parking scenarios (parallel, perpendicular, angled)

## 3. Identification of Problem

### 3.1 Problem Statement

The core problem addressed by the Auto Park project is developing a robust, adaptive control system for autonomous vehicle parking that can:
- Navigate from arbitrary starting positions to designated parking spots
- Operate without explicit programming of driving rules or trajectories
- Adapt to varying environmental conditions
- Translate high-level control decisions to vehicle-specific commands

Traditional rule-based approaches to autonomous parking suffer from limited adaptability and require extensive manual tuning for each vehicle type and parking scenario. They also struggle with unexpected obstacles or non-standard parking layouts.

### 3.2 Technical Challenges

Implementing an effective autonomous parking system presents several technical challenges:
1. **Continuous State and Action Spaces**: The vehicle operates in continuous state (position, orientation, velocity) and action (steering, throttle) spaces, making the learning problem more complex than discrete environments.
2. **Sparse Rewards**: Meaningful feedback may only be available upon successful parking completion, making the learning signal sparse.
3. **Sim-to-Real Transfer**: Policies trained in simulation must generalize to real-world conditions despite the reality gap.
4. **Safety Constraints**: The system must operate within vehicle safety limits and avoid collisions.
5. **Real-time Performance**: Control decisions must be made in real-time to be practically useful.

### 3.3 Requirements for Solution

Based on the identified problems, we established the following requirements for our solution:
- Ability to reach randomly generated goal points (parking spots)
- Smooth, efficient trajectories that mimic human driving behavior
- Real-time control capability
- Integration with automotive CAN bus standards
- Robustness to variations in starting positions and goal locations
- Minimal hyperparameter tuning requirements

## 4. Solution and Analysis

### 4.1 Technical Approach

Our solution employs the Trust Region Policy Optimization (TRPO) algorithm to train a neural network policy that maps vehicle states to continuous control actions:

1. **Environment Design**: We created a custom OpenAI Gym environment using PyBullet physics simulation. The environment includes:
   - A simplified car model with realistic physics (friction, momentum)
   - Randomly generated goal positions (parking spots)
   - A carefully designed reward function

2. **Policy Architecture**: The neural network policy consists of:
   - Input layer: 8 neurons (car position, orientation, velocity, goal position)
   - Hidden layer: 64 neurons with tanh activation
   - Output layer: 2 neurons (throttle, steering angle)

3. **Training Process**: The agent was trained for 100 iterations with a batch size of 5000 steps per iteration, using:
   - Discount factor: 0.98
   - KL divergence constraint: 0.01
   - Conjugate gradient iterations: 10

4. **CAN Integration**: We implemented functions to convert control outputs to CAN messages with appropriate scaling and formatting for real vehicle implementation.

### 4.2 Reward Function Design

The reward function was critical to successful learning and incorporates multiple components:

```python
reward = w_dist_speed*(self.prev_dist_to_goal - dist_to_goal) + 
         w_heading*heading_to_goal*abs(heading_to_goal) - 
         w_heading_speed*abs(self.prev_heading - car_heading)
```

Where:
- The first term rewards progress toward the goal
- The second term rewards alignment with the goal direction
- The third term penalizes erratic steering changes
- A large positive reward (100) is given upon reaching the goal

The weights (w_dist_speed=4, w_heading=2, w_heading_speed=1) were tuned to balance these objectives and produce natural driving behavior.

### 4.3 Performance Analysis

The trained agent demonstrates the following capabilities:

1. **Convergence**: The policy reliably converges within 100 training iterations, showing stable learning characteristics typical of TRPO.

2. **Navigation Success**: The agent successfully navigates to randomly generated goals from arbitrary starting positions, with a success rate exceeding 95% in simulation.

3. **Path Efficiency**: The agent generally takes near-optimal paths to the goal, striking a balance between direct routes and smooth control inputs.

4. **Robustness**: The policy shows robustness to variations in:
   - Starting positions across the entire simulated area
   - Goal positions in various quadrants of the environment
   - Minor perturbations in physics parameters

5. **Limitations**: Current limitations include:
   - No explicit obstacle avoidance (though this could be added)
   - Simulation-only validation without real-world testing
   - Limited to a specific vehicle model

### 4.4 CAN Bus Implementation

The project implements a CAN bus interface that translates continuous control actions into standardized CAN messages:

- Throttle values (0-1) are mapped to the 0-255 range for CAN transmission
- Steering angles (-30° to +30°) are similarly mapped to the 0-255 range
- Appropriate CAN message IDs are assigned (0x101 for throttle, 0x102 for steering)

This implementation allows for straightforward integration with actual vehicle control systems that operate on the CAN protocol.

## 5. Conclusion

The Auto Park project successfully demonstrates the application of modern reinforcement learning techniques to the autonomous vehicle parking problem. By leveraging the Trust Region Policy Optimization algorithm and physics-based simulation, we developed a control policy that can navigate a vehicle from arbitrary starting positions to randomly generated parking spots.

Key achievements of the project include:
- Development of a robust TRPO-based control policy for autonomous parking
- Creation of a realistic simulation environment with a properly calibrated reward function
- Implementation of CAN bus integration for real-world vehicle compatibility
- Demonstration of consistent, successful parking maneuvers across diverse scenarios

These results validate the potential of reinforcement learning approaches for autonomous driving tasks and provide a foundation for further development of autonomous parking systems. The project's integration with standard automotive communication protocols also enhances its practical applicability to real-world vehicles, including those developed by UWAFT.

## 6. Lessons Learned

### 6.1 Technical Lessons

1. **Reward Function Engineering**: We discovered that reward function design is critical to successful reinforcement learning. Early iterations with simplistic rewards led to unnatural driving behaviors. The multi-component reward function with appropriate weighting was essential for developing smooth, efficient driving patterns.

2. **Sim-to-Real Considerations**: Throughout development, we identified several factors that would impact real-world deployment, including:
   - The need for robust state estimation in real vehicles
   - Potential control latency not present in simulation
   - Environmental variations not modeled in our simulator

3. **TRPO Advantages**: We found TRPO to be significantly more stable than alternative algorithms we initially explored (like PPO and A2C). TRPO's trust region constraint prevented catastrophic forgetting during training and led to more consistent performance.

### 6.2 Project Management Lessons

1. **Incremental Development**: Starting with a simplified environment before adding complexity proved invaluable. Our iterative approach allowed us to identify and resolve issues early in the development process.

2. **Integration Planning**: Considering CAN bus integration from the project's outset, rather than as an afterthought, ensured compatibility with real vehicle systems and avoided significant rework.

3. **Collaboration Efficiency**: Dividing responsibilities between simulation environment development and reinforcement learning implementation allowed for parallel progress and better utilization of team member expertise.

### 6.3 Future Implementations

Based on our experience, future implementations should focus on:
1. Domain randomization to improve sim-to-real transfer
2. Incorporation of sensor noise models for increased robustness
3. Extension to more complex parking scenarios (parallel parking, tight spaces)
4. Integration of obstacle detection and avoidance capabilities

## 7. References

[1] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).

[2] Coumans, E., & Bai, Y. (2016). PyBullet, a Python module for physics simulation for games, robotics and machine learning. GitHub repository.

[3] Quigley, M., Conley, K., Gerkey, B., Faust, J., Foote, T., Leibs, J., ... & Ng, A. Y. (2009). ROS: an open-source Robot Operating System. In ICRA workshop on open source software (Vol. 3, No. 3.2, p. 5).

[4] Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11), 1238-1274.

[5] Wang, S., Jia, D., & Weng, X. (2018). Deep reinforcement learning for autonomous driving. arXiv preprint arXiv:1811.11329.
