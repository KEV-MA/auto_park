import gym
import numpy as np
import math
import pybullet as p
from sim_env.car import Car
from sim_env.plane import Plane
from sim_env.goal import Goal
from sim_env.obstacle import Obstacle
import matplotlib.pyplot as plt

class SimpleDrivingEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, use_gui=False, num_obstacles=5):
		#super(SimpleDrivingEnv, self).__init__()
		self.action_space = gym.spaces.box.Box(low=np.array([0, -0.6], dtype=np.float32), high=np.array([1, 0.6], dtype=np.float32))
		self.observation_space = gym.spaces.box.Box(low=np.array([-10, -10, -1, -1, -5, -5, -10, -10, -10, -10], dtype=np.float32), high=np.array([10, 10, 1, 1, 5, 5, 10, 10, 10, 10], dtype=np.float32))
		self.np_random, _ = gym.utils.seeding.np_random()

		self.client = p.connect(p.GUI if use_gui else p.DIRECT)
		# Reduce length of episodes for RL algorithms
		p.setTimeStep(1/30, self.client)

		self.car = None
		self.goal = None
		self.obstacles = []
		self.num_obstacles = num_obstacles
		self.done = False
		self.prev_dist_to_goal = None
		self.prev_orientation = None
		self.rendered_img = None
		self.render_rot_matrix = None
		self.reset()

	def step(self, action):
		# Feed action to the car and get observation of car's state
		self.car.apply_action(action)
		p.stepSimulation()
		car_ob = self.car.get_observation()

		# Compute reward as weighted sum of speed, heading, distance to goal
		dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 + (car_ob[1] - self.goal[1]) ** 2))
		car_vel_vector = [car_ob[4], car_ob[5]]
		goal_traj_vector = [self.goal[0]-car_ob[0], self.goal[1]-car_ob[1]]
		heading_to_goal = np.dot(car_vel_vector / np.linalg.norm(car_vel_vector), goal_traj_vector / np.linalg.norm(goal_traj_vector))
		car_heading = math.atan2(car_ob[3], car_ob[2])
		
		# Check for collisions with obstacles
		car_id, _ = self.car.get_ids()
		for obstacle in self.obstacles:
			obstacle_id, _ = obstacle.get_ids()
			if len(p.getContactPoints(car_id, obstacle_id)) > 0:
				self.done = True
				obs = list(car_ob) + list(self.goal) + self._get_closest_obstacle(car_ob)
				return np.array(obs, dtype=np.float32), -100, self.done, dict()
		
		w_dist_speed = 4
		w_heading_speed = 1
		w_heading = 2
		reward = w_dist_speed*(self.prev_dist_to_goal - dist_to_goal) + w_heading*heading_to_goal*abs(heading_to_goal) - w_heading_speed*abs(self.prev_heading - car_heading)
		
		self.prev_dist_to_goal = dist_to_goal
		self.prev_heading = car_heading

		# Done by running off boundaries
		if (car_ob[0] >= 10 or car_ob[0] <= -10 or car_ob[1] >= 10 or car_ob[1] <= -10):
			self.done = True
		# Done by reaching goal
		elif dist_to_goal < 1:
			self.done = True
			reward = 100

		obs = list(car_ob) + list(self.goal) + self._get_closest_obstacle(car_ob)
		return np.array(obs, dtype=np.float32), reward, self.done, dict()

	def _get_closest_obstacle(self, car_pos):
		closest_dist = float('inf')
		closest_pos = [0, 0]
		
		for obstacle in self.obstacles:
			obstacle_id, _ = obstacle.get_ids()
			pos = p.getBasePositionAndOrientation(obstacle_id)[0]
			dist = math.sqrt((car_pos[0] - pos[0])**2 + (car_pos[1] - pos[1])**2)
			if dist < closest_dist:
				closest_dist = dist
				closest_pos = [pos[0], pos[1]]
		
		return closest_pos

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		p.resetSimulation(self.client)
		p.setGravity(0, 0, -10)
		
		# Reload the plane and car
		Plane(self.client)
		self.car = Car(self.client)

		# Set the goal to a random target
		x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else self.np_random.uniform(-9, -5))
		y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else self.np_random.uniform(-9, -5))
		self.goal = (x, y)
		self.done = False

		# Visual element of the goal
		Goal(self.client, self.goal)

		# Add random obstacles
		self.obstacles = []
		for _ in range(self.num_obstacles):
			while True:
				x = self.np_random.uniform(-8, 8)
				y = self.np_random.uniform(-8, 8)
				# Check if position is far enough from car start and goal
				if math.sqrt(x*x + y*y) > 2 and math.sqrt((x-self.goal[0])**2 + (y-self.goal[1])**2) > 2:
					self.obstacles.append(Obstacle(self.client, (x, y)))
					break

		# Get observation to return
		car_ob = self.car.get_observation()

		self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 + (car_ob[1] -self.goal[1]) ** 2))
		self.prev_heading = math.atan2(car_ob[3], car_ob[2])
		obs = list(car_ob) + list(self.goal) + self._get_closest_obstacle(car_ob)
		return np.array(obs, dtype=np.float32)

	def render(self, mode='human'):
		if self.rendered_img is None:
			self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

		# Base information
		car_id, client_id = self.car.get_ids()
		proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100)
		pos, ori = [list(l) for l in p.getBasePositionAndOrientation(car_id, client_id)]
		pos[2] = 0.2

		# Rotate camera direction
		rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
		camera_vec = np.matmul(rot_mat, [1, 0, 0])
		up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
		view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

		# Display image
		frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
		frame = np.reshape(frame, (100, 100, 4))
		self.rendered_img.set_data(frame)
		plt.draw()
		plt.pause(.00001)

	def close(self):
		p.disconnect(self.client)


