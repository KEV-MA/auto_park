import time
import argparse
import os
import numpy as np
import gym
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

LOG_DIR = "./logs/"
FRAMERATE = 30

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

def main():
	parser = argparse.ArgumentParser(description='Auto Park Agent')
	parser.add_argument('--run-only', action='store_true', help='Only run the trained agent without training')
	parser.add_argument('--gui', action='store_true', help='Enable GUI visualization mode')
	args = parser.parse_args()
	
	# create callback for training
	callback = TrainAndLoggingCallback(check_freq=100000, save_path="./")

	# register environment
	gym.envs.registration.register(
		id="ParkingEnv-v0",
		entry_point="sim_env.simple_driving_env:SimpleDrivingEnv",
		kwargs={"use_gui": args.gui}
	)
	
	# create agent and environment
	env = gym.make("ParkingEnv-v0")
	agent = None
	
	# generate paths for training
	env.save_paths()
	
	if not args.run_only:
		"""
		agent = PPO.load("agent_park_ppo_v5_dyn")
		agent = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.001, n_steps=500)
		agent.learn(total_timesteps=2000000, callback=callback)
		agent.save("agent_park_ppo")
		"""
		n_actions = env.action_space.shape[-1]
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
		agent = TD3.load("best_model_700000.zip")
		agent = TD3("MlpPolicy", env, tensorboard_log=LOG_DIR, verbose=1, action_noise=action_noise, learning_rate=0.001, buffer_size=1000000, gamma=0.95, batch_size=256, train_freq=(15, "step"), policy_kwargs=dict(net_arch=[256, 256, 256]))
		agent.learn(total_timesteps=1000000, callback=callback)
		agent.save("agent_park_td3_v0")
	else:
		agent = TD3.load("best_model_900000.zip")
	
	# reset environment
	ob = env.reset()
	
	# test agent
	while True:
		action, _states = agent.predict(ob)
		ob, reward, done, info = env.step(action)
		env.render()
		if done:
			ob = env.reset()
			time.sleep(1/FRAMERATE)

if __name__ == '__main__':
	main()
