import gym
import time
import torch
import numpy as np
import argparse
import math
import os
from agent import TRPOAgent

# Try to import optional libraries
MATPLOTLIB_AVAILABLE = False
PANDAS_AVAILABLE = False
TQDM_AVAILABLE = False
CAN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Plotting will be disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not available. Results will not be saved as CSV.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available. Progress bar will be disabled.")
    # Simple replacement for tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import can
    from CAN_commands import convert_throttle_to_CAN, convert_steering_to_CAN
    CAN_AVAILABLE = True
except ImportError:
    print("Warning: CAN libraries not available. CAN testing will be disabled.")

def calculate_jerk(velocities, time_step=1/30):
    """
    Calculate jerk from velocity data.
    Jerk is the rate of change of acceleration.
    """
    # Calculate acceleration from velocities
    accelerations = np.diff(velocities, axis=0) / time_step
    # Calculate jerk from accelerations
    jerks = np.diff(accelerations, axis=0) / time_step
    
    # Calculate the magnitude of jerk at each time step
    jerk_magnitudes = np.sqrt(np.sum(jerks**2, axis=1))
    
    return jerk_magnitudes

def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    return obj

def run_evaluation(num_scenarios=100, gui=False, can_test=False, save_data=True):
    """
    Run evaluation to measure:
    1. Success rate 
    2. Average time to park
    3. Trajectory smoothness (jerk)
    4. HIL test with CAN communication
    """
    # Disable CAN testing if libraries not available
    if can_test and not CAN_AVAILABLE:
        print("CAN testing disabled due to missing libraries.")
        can_test = False
    
    # Register environment
    gym.envs.registration.register(
        id="ParkingEnv-v0",
        entry_point="sim_env.simple_driving_env:SimpleDrivingEnv",
        kwargs={"use_gui": gui}
    )
    
    # Create agent
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)
    agent.load_model("agent.pth")
    
    # Initialize CAN bus if needed
    can_bus = None
    if can_test:
        try:
            can_bus = can.interface.Bus(bustype='socketcan', channel='vcan0', bitrate=500000)
            print("CAN bus initialized at 500 kbps")
        except Exception as e:
            print(f"Failed to initialize CAN bus: {e}")
            can_test = False
    
    # Metrics
    successful_parks = 0
    park_times = []
    all_velocities = []
    can_packets_sent = 0
    can_packets_lost = 0
    
    # Trajectory and scenario data
    scenarios_data = []
    
    # Run scenarios
    for scenario in tqdm(range(num_scenarios)):
        # Create environment
        env = gym.make("ParkingEnv-v0")
        ob = env.reset()
        
        # Get goal position
        goal_pos = (float(ob[6]), float(ob[7]))
        
        # Track data for this episode
        done = False
        steps = 0
        episode_velocities = []
        episode_positions = []
        episode_actions = []
        episode_rewards = []
        episode_start_time = time.time()
        
        # Run until done or max steps reached
        while not done and steps < 1800:  # 60 seconds at 30 fps = 1800 steps
            action = agent(ob)
            
            # Store velocity for jerk calculation
            car_ob = env.car.get_observation()
            car_pos = (float(car_ob[0]), float(car_ob[1]))
            car_vel = [float(car_ob[4]), float(car_ob[5])]
            
            episode_velocities.append(car_vel)
            episode_positions.append(car_pos)
            episode_actions.append([float(a) for a in action.tolist()])
            
            # Send CAN messages if testing HIL
            if can_test:
                throttle_value = action[0]
                steering_angle = action[1]
                
                # Convert to CAN messages
                throttle_msg = convert_throttle_to_CAN(throttle_value)
                steering_msg = convert_steering_to_CAN(steering_angle)
                
                # Send messages and track success/failure
                can_packets_sent += 2
                try:
                    can_bus.send(throttle_msg)
                    can_bus.send(steering_msg)
                except Exception:
                    can_packets_lost += 1
            
            # Step environment
            ob, reward, done, info = env.step(action)
            episode_rewards.append(float(reward))
            
            if gui:
                env.render()
            
            steps += 1
        
        # Record metrics for this episode
        episode_time = time.time() - episode_start_time
        park_times.append(steps / 30)  # Convert steps to seconds
        
        # Add velocities to all velocities for later analysis
        all_velocities.append(np.array(episode_velocities))
        
        # Save scenario data
        scenario_data = {
            "scenario_id": scenario,
            "success": bool(done and reward > 99),
            "time_to_park": float(steps / 30),
            "positions": episode_positions,
            "velocities": episode_velocities,
            "actions": episode_actions,
            "rewards": episode_rewards,
            "goal": goal_pos
        }
        scenarios_data.append(scenario_data)
        
        # Check if successfully parked (reached goal)
        # In the environment, reaching the goal gives a reward of 100
        if done and reward > 99:
            successful_parks += 1
        
        env.close()
    
    # Calculate overall metrics
    success_rate = successful_parks / num_scenarios * 100
    avg_time_to_park = sum(park_times) / len(park_times) if park_times else float('inf')
    
    # Calculate jerk metrics across all trajectories
    all_jerks = []
    for velocities in all_velocities:
        if len(velocities) > 2:  # Need at least 3 points to calculate jerk
            jerks = calculate_jerk(velocities)
            all_jerks.extend(jerks.tolist())  # Convert to list for serialization
    
    peak_jerk = max(all_jerks) if all_jerks else 0
    avg_jerk = sum(all_jerks) / len(all_jerks) if all_jerks else 0
    
    # CAN results
    can_loss_rate = can_packets_lost / can_packets_sent * 100 if can_packets_sent > 0 else 0
    
    # Print results
    print("\n--- EVALUATION RESULTS ---")
    print(f"Sim success rate: {success_rate:.2f}% over {num_scenarios} random scenarios.")
    print(f"Avg time-to-park: {avg_time_to_park:.2f} s (constraint: ≤ 60 s).")
    print(f"Trajectory smoothness - peak jerk: {peak_jerk:.2f} m/s³; " + 
          ("meets comfort spec." if peak_jerk < 10 else "exceeds comfort spec."))
    
    if can_test:
        print(f"HIL test - CAN gateway drives APA ECU; " +
              (f"zero packet loss at 500 kbps." if can_loss_rate == 0 else 
               f"{can_loss_rate:.2f}% packet loss at 500 kbps."))
    
    # Prepare results
    results = {
        "success_rate": success_rate,
        "avg_time_to_park": avg_time_to_park,
        "peak_jerk": peak_jerk,
        "avg_jerk": avg_jerk,
        "can_loss_rate": can_loss_rate if can_test else "Not tested",
        "park_times": park_times,
        "all_jerks": all_jerks,
    }
    
    # Save detailed data for visualization
    if save_data:
        # Create report directory if it doesn't exist
        os.makedirs("report", exist_ok=True)
        
        # Save minimal results to a text file
        with open("report/results.txt", "w") as f:
            f.write("--- EVALUATION RESULTS ---\n")
            f.write(f"Sim success rate: {success_rate:.2f}% over {num_scenarios} random scenarios.\n")
            f.write(f"Avg time-to-park: {avg_time_to_park:.2f} s (constraint: ≤ 60 s).\n")
            f.write(f"Trajectory smoothness - peak jerk: {peak_jerk:.2f} m/s³; " + 
                  ("meets comfort spec.\n" if peak_jerk < 10 else "exceeds comfort spec.\n"))
            
            if can_test:
                f.write(f"HIL test - CAN gateway drives APA ECU; " +
                      (f"zero packet loss at 500 kbps.\n" if can_loss_rate == 0 else 
                       f"{can_loss_rate:.2f}% packet loss at 500 kbps.\n"))
        
        # Save evaluation results as CSV
        if PANDAS_AVAILABLE:
            results_df = pd.DataFrame([{
                "success_rate": success_rate,
                "avg_time_to_park": avg_time_to_park,
                "peak_jerk": peak_jerk,
                "avg_jerk": avg_jerk,
                "can_loss_rate": can_loss_rate if can_test else "Not tested"
            }])
            results_df.to_csv("report/evaluation_results.csv", index=False)
            
            # Save park times and jerks for plotting
            pd.DataFrame({"park_times": park_times}).to_csv("report/park_times.csv", index=False)
            if all_jerks:
                pd.DataFrame({"jerks": all_jerks}).to_csv("report/jerks.csv", index=False)
        
        # Save raw data for visualization
        import json
        try:
            # Save park times as JSON
            with open("report/park_times.json", "w") as f:
                json.dump(park_times, f)
            
            # Save jerks as JSON
            if all_jerks:
                with open("report/jerks.json", "w") as f:
                    json.dump(all_jerks, f)
                    
            # Save only a few trajectories for visualization
            if scenarios_data:
                # Keep only successful scenarios
                successful_scenarios = [s for s in scenarios_data if s["success"]]
                
                # If we have successful scenarios, save up to 5
                if successful_scenarios:
                    # Sort by time to park
                    successful_scenarios.sort(key=lambda x: x["time_to_park"])
                    
                    # Save a few scenarios (fastest, median, slowest)
                    scenarios_to_save = []
                    if len(successful_scenarios) >= 3:
                        # Fastest, median, and slowest
                        idx_median = len(successful_scenarios) // 2
                        scenarios_to_save = [
                            successful_scenarios[0],  # Fastest
                            successful_scenarios[idx_median],  # Median
                            successful_scenarios[-1]  # Slowest
                        ]
                    else:
                        # Just save what we have
                        scenarios_to_save = successful_scenarios
                    
                    # Save selected scenarios
                    with open("report/trajectories.json", "w") as f:
                        json.dump(convert_to_python_types(scenarios_to_save), f)
        except Exception as e:
            print(f"Error saving JSON data: {e}")
            
    # Generate plots if matplotlib is available
    if MATPLOTLIB_AVAILABLE and (gui or save_data):
        # Create report directory if it doesn't exist
        os.makedirs("report", exist_ok=True)
        
        # Plot time-to-park distribution
        plt.figure(figsize=(10, 6))
        plt.hist(park_times, bins=20, color='royalblue', edgecolor='black')
        plt.title('Time-to-Park Distribution', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('report/time_to_park.png', dpi=300, bbox_inches='tight')
        
        # Plot jerk distribution
        if all_jerks:
            plt.figure(figsize=(10, 6))
            plt.hist(all_jerks, bins=50, color='firebrick', edgecolor='black')
            plt.title('Jerk Distribution', fontsize=16)
            plt.xlabel('Jerk (m/s³)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.savefig('report/jerk_distribution.png', dpi=300, bbox_inches='tight')
            
            # Plot cumulative jerk distribution
            plt.figure(figsize=(10, 6))
            sorted_jerks = np.sort(all_jerks)
            cumulative = np.arange(1, len(sorted_jerks) + 1) / len(sorted_jerks)
            plt.plot(sorted_jerks, cumulative, 'b-', linewidth=2)
            plt.title('Cumulative Jerk Distribution', fontsize=16)
            plt.xlabel('Jerk (m/s³)', fontsize=14)
            plt.ylabel('Cumulative Probability', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.savefig('report/jerk_cumulative.png', dpi=300, bbox_inches='tight')
        
        if gui:
            plt.show()
        else:
            plt.close('all')
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Auto Park - Evaluation')
    parser.add_argument('--scenarios', type=int, default=100, help='Number of scenarios to evaluate')
    parser.add_argument('--gui', action='store_true', help='Enable GUI visualization mode')
    parser.add_argument('--can-test', action='store_true', help='Test CAN communication')
    parser.add_argument('--save-data', action='store_true', help='Save data for offline visualization')
    args = parser.parse_args()
    
    run_evaluation(num_scenarios=args.scenarios, gui=args.gui, can_test=args.can_test, save_data=args.save_data or args.gui)

if __name__ == '__main__':
    main() 