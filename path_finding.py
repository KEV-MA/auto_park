import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import rsplan
#import turtle
#import reeds_shepp_algorithm.utils as utils
#import reeds_shepp_algorithm.draw as draw
#import reeds_shepp_algorithm.reeds_shepp as reeds_shepp
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot

def _plot_arrow(
    x: float,
    y: float,
    yaw: float,
    length: float = 0.5,
    width: float = 0.3,
    label: str = "",
) -> None:
    """Adds an arrow to the plot."""
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=width,
        head_length=width,
    )
    plt.plot(x, y, marker="s", label=label)

def _viz_path(rs_path: rsplan.Path, path_num: int) -> None:
    """Visualizes the given path in the plot."""
    x_coords, y_coords, _ = rs_path.coordinates_tuple()
    path_plt = plt.subplot(111)
    path_plt.plot(x_coords, y_coords, label=(f"Path ix: {path_num}"))

    # Shrink current axis's height by 10% on the bottom
    box = path_plt.get_position()
    path_plt.set_position(
        [box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95]
    )

    # Put a legend below current axis
    path_plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=3
    )
    plt.show(block=False)

def find_car_path(start, goal, obstacles=None, show_rrt=False, show_reeds=False):
	# define search space dimensions
	"""
	X_dimensions = np.array([(0, 100), (0, 100)])
	obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80),
	                     (60, 20, 80, 40), (60, 60, 80, 80)])
	
	X_dimensions = np.array([(-10, 10), (-10, 10)])
	X = SearchSpace(X_dimensions, obstacles)
	
	x_init = (start[0], start[1])  # starting location
	x_goal = (goal[0], goal[1])  # goal location

	q = 8  # length of tree edges
	r = 0.1  # length of smallest edge to check for intersection with obstacles
	max_samples = 1024  # max number of samples to take before timing out
	rewire_count = 32  # optional, number of nearby branches to rewire
	prc = 0.1  # probability of checking for a connection to goal

	# create rrt_search
	path = None
	while not path:
		rrt = RRTStar(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
		path = rrt.rrt_star()

	if show_rrt:
		plot = Plot("rrt_star_2d")
		plot.plot_tree(X, rrt.trees)
		if path is not None:
		    plot.plot_path(X, path)
		plot.plot_obstacles(X, obstacles)
		plot.plot_start(X, x_init)
		plot.plot_goal(X, x_goal)
		plot.draw(auto_open=True)

	# add orientation to path
	path_corrected = []
	path_corrected.append((path[0][0], path[0][1], start[2]))
	for i in range(len(path) - 2,len(path) - 1):
		dx = path[i+1][0] - path[i][0]
		dy = path[i+1][1] - path[i][1]
		theta = math.atan2(dy, dx)
		path_corrected.append((path[i][0], path[i][1], theta))
	path_corrected.append((path[-1][0], path[-1][1], goal[2]))
	"""
	# optimize path for car using reeds shepp curves
	turn_radius = 4
	runway_length = 0.1
	step_size = 0.5
	#rs_path = []
	"""
	for i in range(len(path_corrected)-1):
		generated_path = rsplan.path(path_corrected[i], path_corrected[i+1], turn_radius, runway_length, step_size)
		rs_path.append(generated_path)
		
		if show_reeds:
			_plot_arrow(*path_corrected[i])
			_viz_path(generated_path, i+1)
			_plot_arrow(*path_corrected[i+1])
	"""
	rs_path = rsplan.path(start, goal, turn_radius, runway_length, step_size)
	if show_reeds:
		_plot_arrow(*start)
		_viz_path(rs_path, -1)
		_plot_arrow(*goal)
	if show_reeds:
		plt.grid()
		plt.show()
		
	return rs_path

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Auto Park Path Finding")
	parser.add_argument("--show-rrt", action="store_true", help="Show rrt generated path")
	parser.add_argument("--show-reeds", action="store_true", help="Show reed shepp refined path")
	args = parser.parse_args()
	
	car_length = 1
	car_width = 1
	n_cars = 2
	#x = (np.random.uniform(5, 9) if np.random.randint(2) else np.random.uniform(-9, -5))
	#y = (np.random.uniform(5, 9) if np.random.randint(2) else np.random.uniform(-9, -5))
	goal = (-1, 3, 0)
	start = (0, 0, np.pi/2)
	obstacles = np.array([
		(goal[0]-car_length/2-car_length, goal[1]-car_width/2-car_width*n_cars, goal[0]-car_length/2, goal[1]+car_width/2+car_width*n_cars),
		(goal[0]-car_length/2, goal[1]-car_width/2-car_width*n_cars, goal[0]+car_length/2, goal[1]-car_width/2),
		(goal[0]-car_length/2, goal[1]+car_width/2, goal[0]+car_length/2, goal[1]+car_width/2+car_width*n_cars),
		(goal[0]+2*car_length, goal[1]-car_width/2-car_width*n_cars, goal[0]+3*car_length, goal[1]+car_width/2+car_width*n_cars)
		])
	rs_path = find_car_path(start=start, goal=goal, obstacles=obstacles, show_rrt=args.show_rrt, show_reeds=args.show_reeds)
	print(len(rs_path.coordinates_tuple()[0]))
	print(rs_path.coordinates_tuple()[0])
	print(rs_path.coordinates_tuple()[1])
	print(rs_path.coordinates_tuple()[2])
	
"""
# get all reeds paths
all_paths = []
for i in range(len(path_corrected) - 1):
	all_paths.append(reeds_shepp.get_all_paths(path_corrected[i], path_corrected[i+1]))

# get optimal path
optimal_path = []
path_length = 0
for i in range(len(path_corrected) - 1):
	optimal_path.append(reeds_shepp.get_optimal_path(path_corrected[i], path_corrected[i+1]))
	path_length += reeds_shepp.path_length(optimal_path[i])
print("Shortest path length: {} px.".format(int(draw.scale(path_length))))
print(optimal_path)

if args.show_reeds:
	# initialize turtle
	screen = turtle.Screen()
	screen.setworldcoordinates(0, 0, 150, 150)
	car = turtle.Turtle()
	car.speed(0) # 0: fast; 1: slow, 8.4: cool
	car.shape('arrow')
	car.resizemode('user')
	car.shapesize(1, 1)
	
	# draw vectors representing points in path_corrected
	for pt in path_corrected:
		draw.goto(car, pt)
		draw.vec(car)

	# draw all routes found
	for i in range(len(path_corrected) - 1):
		paths = reeds_shepp.get_all_paths(path_corrected[i], path_corrected[i+1])
		for path in paths:
		    draw.set_random_pencolor(car)
		    draw.goto(car, path_corrected[i])
		    draw.draw_path(car, path)

	# draw shortest route
	car.pencolor(1, 0, 0)
	car.pensize(3)
	car.speed(1)
	while True:
		car.hideturtle()
		draw.goto(car, path_corrected[0])
		car.showturtle()
		for i in range(len(path_corrected) - 1):
			path = reeds_shepp.get_optimal_path(path_corrected[i], path_corrected[i+1])
			draw.draw_path(car, path)
		car.speed(4)
	turtle.done()
"""

