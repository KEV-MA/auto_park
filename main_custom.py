import gym
import time
import torch
import argparse
import os
from agent import TRPOAgent

def main():
	parser = argparse.ArgumentParser(description='Auto Park - TRPO Agent')
	parser.add_argument('--run-only', action='store_true', help='Only run the trained agent without training')
	parser.add_argument('--gui', action='store_true', help='Enable GUI visualization mode')
	args = parser.parse_args()

	# register environment
	gym.envs.registration.register(
		id="ParkingEnv-v0",
		entry_point="sim_env.simple_driving_env:SimpleDrivingEnv",
		kwargs={"use_gui": args.gui}
	)
	
	# create agent and environment
	nn = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.Tanh(),
		     torch.nn.Linear(64, 2))	# set inputs
	agent = TRPOAgent(policy=nn)
	
	agent.load_model("./checkpoints/agent_park_trpo_v2_r2_dyn.pth")
	if not args.run_only:
		agent.train("ParkingEnv-v0", seed=0, batch_size=5000, iterations=100, max_episode_length=250, verbose=True)
		agent.save_model("agent_park_trpo_v3.pth")
	
	# create environment
	env = gym.make("ParkingEnv-v0")
	
	# generate paths
	env.save_paths()
	
	# reset environment
	ob = env.reset()
	
	# test agent
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
