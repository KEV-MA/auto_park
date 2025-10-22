import gym
import numpy as np
import pybullet as p
from sim_env.car import Car
from sim_env.plane import Plane
from sim_env.goal import Goal
from sim_env.obstacle import Obstacle
from path_finding import find_car_path
from rrt_star_reeds_shepp import RRTStarReedsShepp
import matplotlib.pyplot as plt

class SimpleDrivingEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, use_gui=False):
		#super(SimpleDrivingEnv, self).__init__()
		self.action_space = gym.spaces.box.Box(low=np.array([-0.6, -0.8], dtype=np.float32), high=np.array([0.6, 0.8], dtype=np.float32))
		# car position: (x,y), car orientation: (x,y), car velocity: (x,y), goal position: (x,y,theta), nearest obstacle left, nearest obstacle right
		self.observation_space = gym.spaces.box.Box(low=np.array([-10, -10, -1, -1, -5, -5, -10, -10, -np.pi, 0, -np.pi], dtype=np.float32), high=np.array([10, 10, 1, 1, 5, 5, 10, 10, np.pi, 100, np.pi], dtype=np.float32))
		self.np_random, _ = gym.utils.seeding.np_random()

		self.client = p.connect(p.GUI if use_gui else p.DIRECT)
		
		# Reduce length of episodes for RL algorithms
		p.setTimeStep(1/30, self.client)

		self.car = None
		self.car_width = 0.3
		self.car_length = 0.5
		
		self.goal = None
		self.goals = []
		self.goals_array = []
		self.goals_index = 0
		self.num_paths = 1
		
		self.obstacles = []
		self.map_min = -10
		self.map_max = 10
		self.done = False
		
		self.prev_dist_to_goal = None
		self.prev_heading = None
		
		self.rendered_img = None
		self.render_rot_matrix = None
		#self.reset()

	def step(self, action):
		# Feed action to the car and get observation of car's state
		self.car.apply_action(action)
		p.stepSimulation()
		car_ob = self.car.get_observation()
		"""
		# Compute distance to goal
		dist_to_goal = np.sqrt(((car_ob[0] - self.goal[0]) ** 2 + (car_ob[1] - self.goal[1]) ** 2))
		
		# Compute heading of path
		car_vel_vector = [car_ob[4], car_ob[5]]
		goal_traj_vector = [self.goal[0]-car_ob[0], self.goal[1]-car_ob[1]]
		heading_to_goal = np.dot(car_vel_vector / np.linalg.norm(car_vel_vector), goal_traj_vector / np.linalg.norm(goal_traj_vector))
		car_heading = np.arctan2(car_ob[3], car_ob[2])
		
		w_dist_speed = 4
		w_heading_speed = 1
		w_heading = 2
		reward = w_dist_speed*(self.prev_dist_to_goal - dist_to_goal) + w_heading*heading_to_goal*abs(heading_to_goal) - w_heading_speed*abs(self.prev_heading - car_heading)
		"""
		# Get current goal
		goal = self.goals[self.goal_index]
		
		# Compute distance to nearest goal
		dist_to_goal = np.sqrt(((car_ob[0] - goal[0]) ** 2 + (car_ob[1] - goal[1]) ** 2))
		dist_to_main_goal = np.sqrt(((car_ob[0] - self.goal[0]) ** 2 + (car_ob[1] - self.goal[1]) ** 2))
		
		"""
		# Compute similarity of car velocity direction and shortest path to goal direction
		car_vel_vector = [car_ob[4], car_ob[5]]
		goal_traj_vector = [goal[0]-car_ob[0], goal[1]-car_ob[1]]
		heading_cos = np.dot(car_vel_vector / np.linalg.norm(car_vel_vector), goal_traj_vector / np.linalg.norm(goal_traj_vector))
		"""
		
		# Compute heading to nearest goal
		car_heading = np.arctan2(car_ob[3], car_ob[2])
		angle_to_goal = goal[2] - car_heading
		
		# Compute car velocity to nearest goal
		car_vel = np.sqrt(car_ob[4]**2 + car_ob[5]**2)
		desired_vel = 1
		car_vel_vector = [car_ob[4], car_ob[5]]
		try:
			tmp_goal = self.goals[self.goal_index+1]
			goal_vel_vector = [tmp_goal[0] - goal[0], tmp_goal[1] - goal[1]]
			heading_cos = np.dot(car_vel_vector / np.linalg.norm(car_vel_vector), goal_vel_vector / np.linalg.norm(goal_vel_vector))
		except:
			tmp_goal = self.goals[self.goal_index-1]
			goal_vel_vector = [goal[0] - tmp_goal[0], goal[1] - tmp_goal[1]]
			heading_cos = np.dot(car_vel_vector / np.linalg.norm(car_vel_vector), goal_vel_vector / np.linalg.norm(goal_vel_vector))
		
		w_heading_cos = 3
		w_heading = 1
		w_dist = 1
		w_ori = 1
		w_vel = 3
		w_parking_speed = 12
		
		# immediate rewards = near planned path + speed towards goal + heading similarity - heading speed
		# sparse rewards = reached goal - goal orientation difference - collision
		reward = w_heading_cos*heading_cos*abs(heading_cos) - w_heading*abs(goal[2] - car_heading) - w_dist*dist_to_goal #- w_vel*abs(desired_vel - car_vel)
		
		self.prev_dist_to_goal = dist_to_goal
		self.prev_heading = car_heading
		
		# Done by collisions with obstacles
		for obstacle in self.obstacles:
			contacts = p.getClosestPoints(self.car.car, obstacle.obstacle, distance=0.0)
			if len(contacts) > 0:
				self.done = True
				reward -= 100
		# Done by running off boundaries
		if (car_ob[0] >= 10 or car_ob[0] <= -10 or car_ob[1] >= 10 or car_ob[1] <= -10):
			self.done = True
			reward -= 100
		# Done by reaching main goal
		elif dist_to_main_goal < 0.5:
			self.done = True
			heading_diff = (car_heading - goal[2] + np.pi) % (2 * np.pi) - np.pi
			reward += np.max(100 * (1 - abs(heading_diff) / np.pi) - w_parking_speed*car_vel, 0)
		# Reward for reaching goal
		elif dist_to_goal < 0.5 and goal != self.goal:
			heading_diff = (car_heading - goal[2] + np.pi) % (2 * np.pi) - np.pi
			reward += 100 * (1 - abs(heading_diff) / np.pi)
			self.goal_index += 1
			goal = self.goals[self.goal_index]
			dist_to_goal = np.sqrt(((car_ob[0] - goal[0]) ** 2 + (car_ob[1] - goal[1]) ** 2))
			angle_to_goal = goal[2] - car_heading

		
		"""
		# Get next goal if there is one
		try:
			next_goal = self.goals_array[self.goal_index+1]
		except:
			next_goal = goal
		"""

		ob = np.array(car_ob + goal + (dist_to_goal, angle_to_goal), dtype=np.float32)
		return ob, reward, self.done, dict()

	def generate_path(self, car_start, goal, show_path=True):
		path = []
		
		# Set obstacles around goal
		n_cars = 4
		car_size = 0.6
		space_mult = 6
		tol_mult = 3.5
		
		obstacles_array = []
		
		obstacles_array.append((goal[0]+self.car_length*space_mult, goal[1], car_size))
		obstacles_array.append((goal[0]-self.car_length*1.5, goal[1], car_size))
		for l in range(n_cars):
			obstacles_array.append((goal[0], goal[1]+self.car_width*tol_mult*(l+1), car_size))
			obstacles_array.append((goal[0]+self.car_length*space_mult, goal[1]+self.car_width*tol_mult*(l+1), car_size))
		for r in range(n_cars):
			obstacles_array.append((goal[0], goal[1]-self.car_width*tol_mult*(r+1), car_size))
			obstacles_array.append((goal[0]+self.car_length*space_mult, goal[1]-self.car_width*tol_mult*(r+1), car_size))
		
		# Generate path
		while not path:
			rrt_star_reeds_shepp = RRTStarReedsShepp(car_start, goal, obstacles_array, [self.map_min, self.map_max], max_iter=100)
			path = rrt_star_reeds_shepp.planning(animation=False, search_until_max_iter=True)
		print("Number of Goals:", len(path))
		
		if show_path:
			rrt_star_reeds_shepp.draw_graph()
			plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
			plt.grid(True)
			plt.pause(0.001)
			plt.show()
		
		unique_path = []
		for item in path:
			if item not in unique_path:
				unique_path.append(item)
				
		return unique_path, obstacles_array

	def save_paths(self):
		for i in range(self.num_paths):
			# Set the goal and car to random target
			ran_ori = 0 if self.np_random.integers(2) else 1
			x = (self.np_random.uniform(0,2) if self.np_random.integers(2) else self.np_random.uniform(-2, 0))
			y = (self.np_random.uniform(0,2) if self.np_random.integers(2) else self.np_random.uniform(-2, 0))

			if ran_ori:
				car_start = (0+x, 0+y, np.pi/2)
			else:
				car_start = (0+x, 5+y, -np.pi/2)
			goal = (-1+x, 3+y, 0)
			
			# Generate path based on start and goal points
			path, obstacles_array = self.generate_path(car_start, goal)
			print(path)
			path_ordered = []
			for p in range(len(path)):
				path_ordered.append(tuple(path[len(path)-1-p]))
			self.goals_array.append((car_start, goal, path_ordered, obstacles_array))

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		p.resetSimulation(self.client)
		p.setGravity(0, 0, -9.81)
		
		# Reload the plane
		Plane(self.client)

		# Get random path, start, and goal point
		path_index = self.np_random.integers(self.num_paths)
		car_start = self.goals_array[path_index][0]
		self.goal = self.goals_array[path_index][1]
		self.goals = self.goals_array[path_index][2]
		
		# Load obstacles
		obstacles_array = self.goals_array[path_index][3]
		for ob_pos in obstacles_array:
			self.obstacles.append(Obstacle(self.client, ob_pos))
		
		# Load car and goal
		self.car = Car(self.client, car_start)
		Goal(self.client, self.goal)
		
		self.done = False
		self.goal_index = 0
		
		# Get current goal
		goal = self.goals[self.goal_index]
		
		# Get observation to return
		car_ob = self.car.get_observation()
		
		dist_to_goal = np.sqrt(((car_ob[0] - goal[0]) ** 2 + (car_ob[1] - goal[1]) ** 2))
		angle_to_goal = goal[2] - car_start[2]

		# update previous metrics
		self.prev_dist_to_goal = np.sqrt(((car_ob[0] - goal[0]) ** 2 + (car_ob[1] - goal[1]) ** 2))
		self.prev_heading = np.arctan2(car_ob[3], car_ob[2])
		return np.array(car_ob + goal + (dist_to_goal, angle_to_goal), dtype=np.float32)

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


