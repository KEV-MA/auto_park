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
- Python 3.x
- Git

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
python main.py
```

The script will:
1. Load the pre-trained TRPO agent from `agent.pth`
2. Create a parking environment
3. Run the simulation showing the car navigating to random goal points
4. Display the reward at each step
5. Automatically reset when the goal is reached

## Project Structure

```
auto_park/
├── README.md               # Project documentation
├── agent.py               # TRPO agent implementation
├── agent.pth              # Trained model weights
├── agent_checkpoints/     # Saved model checkpoints
├── main.py               # Main script to run the simulation
├── requirements.txt       # Project dependencies
└── sim_env/              # Simulation environment code
```

## Results

The TRPO model has been successfully trained and demonstrates effective navigation capabilities, consistently reaching random goal points from various starting positions. The trained model weights are stored in `agent.pth`.

## Course Information

This project was developed as part of ME 599 at the University of Waterloo.

## Contributors

Kevin Ma
Arjun Krishna
