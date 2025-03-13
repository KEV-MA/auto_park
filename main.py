import gym
import time
import torch
import argparse
from agent import TRPOAgent
from stable_baselines3 import A2C
from CAN_commands import convert_throttle_to_CAN, convert_steering_to_CAN, print_CAN_message

LOAD_AGENT="parking_agent_v0"

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
	
	# create agent
	nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
		     torch.nn.Linear(64, 2))
	agent = TRPOAgent(policy=nn)

	agent.load_model("agent.pth")

	if not args.run_only:
		agent.train("ParkingEnv-v0", seed=0, batch_size=5000, iterations=100, max_episode_length=250, verbose=True)
		agent.save_model("agent.pth")

	# create environment
	env = gym.make("ParkingEnv-v0")
	ob = env.reset()

	while True:
		action = agent(ob)

	# action array is returned by the agen as [throttle, steering_angle]
		throttle_value = action[0]
		steering_angle = action[1]
		#print(f"Throttle: {throttle_value}, Steering Angle: {steering_angle}")

	# Convert and print the CAN messages
		print(f"Throttle {print_CAN_message(convert_throttle_to_CAN(throttle_value))}, Steering {print_CAN_message(convert_steering_to_CAN(steering_angle))}")

		ob, reward, done, info = env.step(action)
		print("Reward: ",reward)
		env.render()
		if done:
			ob = env.reset()
			time.sleep(1/30)
	"""
	# train agent
	if LOAD_AGENT == "":
		agent = A2C("MlpPolicy", env, verbose=1)
		agent.learn(total_timesteps=1000000)
		agent.save("parking_agent_v0")
	else:
		agent = A2C.load(LOAD_AGENT)
	
	
	# test agent in environment
	while True:
		action, states = agent.predict(ob)
		ob, reward, done, info = env.step(action)
		env.render()
		if done:
			ob = env.reset()
			time.sleep(1/30)
	"""
if __name__ == '__main__':
	main()
