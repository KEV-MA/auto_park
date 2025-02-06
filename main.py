import gym
import time
import torch
from agent import TRPOAgent
from stable_baselines3 import A2C

LOAD_AGENT="parking_agent_v0"

def main():
	# register environment
	gym.envs.registration.register(
		id="ParkingEnv-v0",
		entry_point="sim_env.simple_driving_env:SimpleDrivingEnv",
	)
	
	# create agent
	nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
		     torch.nn.Linear(64, 2))
	agent = TRPOAgent(policy=nn)

	agent.load_model("agent.pth")
	agent.train("ParkingEnv-v0", seed=0, batch_size=5000, iterations=100, max_episode_length=250, verbose=True)
	agent.save_model("agent.pth")

	# create environment
	env = gym.make("ParkingEnv-v0")
	ob = env.reset()

	while True:
		action = agent(ob)
		ob, reward, done, info = env.step(action)
		print(reward)
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
