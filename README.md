# Auto Park System

This repository contains an Autonomous Parking Assist (APA) system using reinforcement learning.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Components

- `main.py`: Main script to train and run the parking agent
- `agent.py`: Implementation of the TRPO reinforcement learning agent
- `sim_env/`: Simulation environment for the parking scenario
- `CAN_commands.py`: Functions to convert control signals to CAN messages
- `evaluate.py`: Evaluation script to measure performance metrics

## Running the Agent

To train and run the agent:
```bash
python main.py
```

To only run a pre-trained agent:
```bash
python main.py --run-only
```

With GUI visualization:
```bash
python main.py --gui
```

## Evaluation

To evaluate the agent's performance and generate metrics:
```bash
python evaluate.py
```

Options for evaluation:
- `--scenarios N`: Number of scenarios to evaluate (default: 100)
- `--gui`: Enable GUI visualization
- `--can-test`: Run HIL test with CAN communication

Example:
```bash
python evaluate.py --scenarios 100 --gui
```

## Performance Metrics

The evaluation script generates the following metrics:

1. Sim success rate - Percentage of successful parking maneuvers over random scenarios
2. Avg time-to-park - Average time in seconds to complete parking
3. Trajectory smoothness - Measures peak jerk (m/s³) for comfort assessment
4. HIL test - Tests CAN communication at 500 kbps

Results are displayed in the console and saved to `evaluation_results.csv`.

## CAN Bus Testing

For CAN bus testing, you need to set up a virtual CAN interface:

```bash
# Load the vcan kernel module
sudo modprobe vcan

# Create a virtual CAN interface
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Run the evaluation with CAN testing
python evaluate.py --can-test
```

## Visualizations

When running with the `--gui` option, the evaluation script generates plots:
- Time-to-park distribution
- Jerk distribution

These plots are saved as PNG files in the project directory.

## Project Overview

This project implements a TRPO (Trust Region Policy Optimization) reinforcement learning algorithm to train a car agent to navigate to randomly generated parking spots. The model has been successfully trained to:
- Navigate from any starting position
- Reach randomly generated goal points
- Avoid obstacles (if present)
- Find optimal paths to the target location

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
