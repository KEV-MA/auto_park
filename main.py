import gym
import time
import torch
import argparse
from agent import PPOAgent  # Use PPOAgent instead of TRPOAgent


def main():
    parser = argparse.ArgumentParser(description='Auto Park - PPO Agent')
    parser.add_argument('--run-only', action='store_true', help='Only run the trained agent without training')
    parser.add_argument('--gui', action='store_true', help='Enable GUI visualization mode')
    args = parser.parse_args()

    # Register environment
    gym.envs.registration.register(
        id="ParkingEnv-v0",
        entry_point="sim_env.simple_driving_env:SimpleDrivingEnv",
        kwargs={"use_gui": args.gui}
    )
    
    # Create agent
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = PPOAgent(policy=nn)
    agent.load_model("agent.pth")  # Load PPO weights from ppo_agent.pth

    # Load pre-trained weights only if --run-only is specified
    if not args.run_only:
        # Train the agent from scratch
        agent.train("ParkingEnv-v0", seed=0, batch_size=5000, iterations=1000, max_episode_length=250, verbose=True)
        agent.save_model("agent.pth")  # Save PPO weights to ppo_agent.pth

    # Create environment
    env = gym.make("ParkingEnv-v0")
    ob = env.reset()

    # Run the agent
    while True:
        action = agent(ob)
        ob, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)

if __name__ == '__main__':
    main()