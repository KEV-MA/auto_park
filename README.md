# Auto Park

A reinforcement learning project for ME 599 at the University of Waterloo that demonstrates autonomous parking capabilities using Trust Region Policy Optimization (TRPO). The model successfully navigates a car from a starting position to randomly generated goal points.

## Project Overview

This project implements a TRPO (Trust Region Policy Optimization) reinforcement learning algorithm to train a car agent to navigate to randomly generated parking spots. The model has been successfully trained to:
- Navigate from any starting position
- Reach randomly generated goal points
- Avoid obstacles (if present)
- Find optimal paths to the target location

## Installation

### Prerequisites
- Python 3

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/auto_park.git
cd auto_park
```

2. Create and activate a virtual environment:
```bash
python -m .venv venv
source .venv/bin/activate  # On Linux/Mac
# or
.\.venv\Scripts\activate  # On Windows
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
The project requires the following main packages:
- gym - For reinforcement learning environments
- pybullet - For physics simulation
- stable-baselines3 - For RL algorithms implementation
- numpy (version 1.23.1) - For numerical computations
- shimmy - For environment compatibility

## Usage

To run the trained model:

```bash
python main.py  # Run with training
python main.py --run-only  # Run without training
```

The script will:
1. Load the pre-trained TRPO agent from `agent.pth`
2. If not using `--run-only`, train the agent for additional iterations
3. Create a parking environment
4. Run the simulation showing the car navigating to random goal points
5. Display the reward at each step
6. Automatically reset when the goal is reached

## Project Structure

```
auto_park/
├── README.md             # Project documentation
├── sim_env/              # Simulation environment code
├── agent_checkpoints/    # Manually archived model checkpoints
├── checkpoints/          # Tensorboard model checkpoints
├── agent.py              # Custom TRPO model implementation
├── agent.pth             # Recently Trained model weights
├── main.py               # Run simulation using stable baseline models
├── main_custom.py        # Run simulation using custom TRPO model
└── requirements.ttx      # Project dependencies
```

## Results

The PPO and TRPO algorithms have been successfully used to train a simple neural network with 
effective navigation capabilities, consistently reaching random goal points from 
various starting positions. The policy creates a trajectory that minimizes distance and 
time travelled to reach the goal. The trained model weights are stored 
in `agent.pth`.  

Instead of the policy planning the trajectory, rrt star with reeds shepp algorithm was used to define a specific path for the car to follow. The policy would be learning how to control the car along that path.  

For these experiments, a policy could be learned to follow a particular path, but attempts to generalize the policy for any path fails even when trained on several dozen paths to random start and goal positions.  

More experimentation with the neural network architecture to generalize a policy to control the car to park independent of the start and goal position is required.

## Contributors

Kevin Ma  
Arjun Krishna
