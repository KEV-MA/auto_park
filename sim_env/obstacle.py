import pybullet as p
import os

class Obstacle:
    def __init__(self, client, position):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        self.obstacle = p.loadURDF(fileName=f_name,
                               basePosition=[position[0], position[1], 0],
                               physicsClientId=client)
        
    def get_ids(self):
        return self.obstacle, self.client 