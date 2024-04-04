#!/usr/bin/python
import sys
import time
import pickle
import numpy as np
import random
import cv2

from itertools import product
from math import cos, sin, atan, atan2, pi, sqrt

import matplotlib
import matplotlib.pyplot as plt

from plotting_utils import draw_plan
from priority_queue import priority_dict

class State:
    """
    2D state.
    """

    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []
        self.cost = 0 # new


    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))

    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x)**2 + (state.y - self.y)**2)

class InformedRRTPlanner:
    """
    Applies the RRT algorithm on a given grid world
    """

    def __init__(self, world, start_state, dest_state):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

        self.start_state = start_state
        self.dest_state = dest_state

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-3:state.y+3, state.x-3:state.x+3] == 0).all()


    def sample_state(self):
        """
        Sample a new FREE state uniformly randomly on the image.
        """
        #TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
        rows, cols = self.world.shape[0], self.world.shape[1]
        state = None

        delta = 10

        x = random.randint(0 + delta, cols - 1 - delta)
        y = random.randint(0 + delta, rows - 1 - delta)
        state = State(x, y, None)

        # Ensures the state returned is free
        while (self.state_is_free(state) == False):
            x = random.randint(0 + delta, cols - 1 - delta)
            y = random.randint(0 + delta, rows - 1 - delta)
            state = State(x, y, None)
        
        return state
    

    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        curr_ptr = state
        path = [state]

        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        # return a reverse copy of the path (so that first state is starting state)
        return path[::-1]


    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state

    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:

        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y

        Otherwise, s_rand is farther than max_radius from s_nearest.
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.

        """

        #TODO: populate x and y properly according to the description above.
        #Note: x and y are integers and they should be in {0, ..., cols -1}
        # and {0, ..., rows -1} respectively
        x = 0
        y = 0

        s_new = State(x, y, s_nearest)

        dist = s_nearest.euclidean_distance(s_rand)

        if (dist < max_radius):
            s_new.x = s_rand.x
            s_new.y = s_rand.y
        else:
            dx = s_rand.x - s_nearest.x
            dy = s_rand.y - s_nearest.y

            # if, for whatever reason, s_nearest and s_rand are the same
            if dx == 0 and dy == 0:
                s_new.x = s_nearest.x
                s_new.y = s_nearest.y

            elif dx == 0:
                s_new.x = s_nearest.x
                s_new.y = s_nearest.y + max_radius

            elif dy == 0:
                s_new.x = s_nearest.x + max_radius
                s_new.y = s_nearest.y

            else:
                angle = atan(dy / dx)

                s_new.x = int(s_nearest.x + round(cos(angle) * max_radius))
                s_new.y = int(s_nearest.y + round(sin(angle) * max_radius))

        return s_new


    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False
        
        # We can either use max_checks:

        max_checks = 100
        # for i in range(max_checks):
        #     ratio = float(i) / max_checks
        #     ...

        # ...or step distance, which I prefer

        step_distance = 30
        total_distance = s_from.euclidean_distance(s_to)
        num_steps = int(total_distance / step_distance)

        # s_from and s_to are basically next to each other
        # we assume s_from and s_to are in a free space, so return True
        # if (num_steps == 0):
        #     return True

        # for i in range(num_steps + 1):  # Include the last point (s_to)
        for i in range(max_checks):
            # ratio = float(i) / num_steps
            ratio = float(i) / max_checks

            x = int(s_from.x + ratio * (s_to.x - s_from.x))
            y = int(s_from.y + ratio * (s_to.y - s_from.y))

            inteprolated_state = State(x, y, s_from)

            if not (self.state_is_free(inteprolated_state)):
                return False

        # Otherwise the line is free, so return true
        return True

    ################################################################

    def near(self, state, tree_nodes, radius):
        """
        Returns all the tree nodes within radius of state.
        """

        near_nodes = []

        for node in tree_nodes:
            if (node.euclidean_distance(state) < radius):
                near_nodes.append(node)
        
        return near_nodes
    

    def RotationToWorldFrame(self, start_state, dest_state, L):
        a1 = np.array([[(dest_state.x - start_state.x) / L],
                       [(dest_state.y - start_state.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C
    

    def get_distance_and_angle(self, start_state, dest_state):
        dx = dest_state.x - start_state.x
        dy = dest_state.y - start_state.y
        return start_state.euclidean_distance(dest_state), atan2(dy, dx)
    

    def init(self, start_state, dest_state):
        cMin, theta = self.get_distance_and_angle(start_state, dest_state)
        C = self.RotationToWorldFrame(start_state, dest_state, cMin)
        xCenter = State(((start_state.x + dest_state.x) / 2), ((start_state.y + dest_state.y) / 2), None)
        x_best = start_state

        return theta, cMin, xCenter, C, x_best
    
    
    def SampleUnitBall(self):
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])
    
    def SampleFreeSpace(self, hybrid_lambda):
    
        hybrid_lambda_r = random.random()
        s_rand = None

        if (hybrid_lambda_r < hybrid_lambda):
            s_rand = dest_state
        else:
            s_rand = self.sample_state()

        return s_rand
    
    def Sample(self, c_max, c_min, x_center, C, hybrid_lambda_r):
        if c_max < np.inf:
            r = [c_max / 2.0,
                 sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            L = np.diag(r)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand_temp = np.dot(np.dot(C, L), x_ball)
                x_rand = State(int(x_rand_temp[(0, 0)] + x_center.x), int(x_rand_temp[(1, 0)] + x_center.y), None)
                if self.state_is_free(x_rand) and 0 <= x_rand.x < len(self.world[0]) and 0 <= x_rand.y < len(self.world):
                    break
        else:
            x_rand = self.SampleFreeSpace(hybrid_lambda_r)
        
        return x_rand
    
    def InGoalRegion(self, node, dest_state, dest_reached_radius):
        if node.euclidean_distance(dest_state) < dest_reached_radius:
            return True

        return False
    
    
    ################################################################

    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # The set containing the nodes of the tree
        tree_nodes = set()
        tree_nodes.add(start_state)

        # image to be used to display the tree
        img = np.copy(self.world)

        plan = [start_state]

        # Variables used for plotting
        num_nodes_list = []  # List to store the number of nodes at each step
        path_length_list = []  # List to store the length of the path at each step

        # hybrid_lambda = random.random()
        hybrid_lambda = 0.3

        theta, dist, x_center, C, x_best = self.init(start_state, dest_state)
        c_best = np.inf

        dist -= dest_reached_radius # needed or we get Math domain error when computing sqrt(c_max ** 2 - c_min ** 2)

        X_soln = set()
        for step in range(max_num_steps):
            num_nodes_list.append(num_nodes_list[step-1] + 1 if len(num_nodes_list) > 0 else 1)
            c_best = float('inf')
            for node in X_soln:
                if node.cost < c_best:
                    print("num nodes", "path cost")
                    print(num_nodes_list[step], node.cost)
                    c_best = node.cost
                    x_best = node

            path_length_list.append(c_best)

            x_rand = self.Sample(c_best, dist, x_center, C, hybrid_lambda)
            x_nearest = self.find_closest_state(tree_nodes, x_rand)
            x_new = self.steer_towards(x_nearest, x_rand, max_steering_radius)

            if self.path_is_obstacle_free(x_nearest, x_new):

                near_nodes = self.near(x_new, tree_nodes, max_steering_radius)
                c_min = x_nearest.cost + x_nearest.euclidean_distance(x_new)

                x_new.parent = x_nearest
                x_new.cost = c_min

                # choose parent
                for x_near in near_nodes:
                    if (self.path_is_obstacle_free(x_near, x_new)):
                        c_new = x_near.cost + x_near.euclidean_distance(x_new)
                        if c_new < c_min:
                            x_new.parent = x_near
                            x_new.cost = c_new
                            c_min = c_new
                
                # rewire
                for x_near in near_nodes:
                    if (self.path_is_obstacle_free(x_near, x_new)):
                        c_near = x_near.cost
                        c_new = x_new.cost + x_near.euclidean_distance(x_new)
                        if c_new < c_near:
                            x_near.parent = x_new
                            x_near.cost = c_new

                tree_nodes.add(x_new)

                if self.InGoalRegion(x_new, dest_state, dest_reached_radius):
                    if self.path_is_obstacle_free(x_new, dest_state):
                        X_soln.add(x_new)

                # plot the new node and edge
                cv2.circle(img, (x_new.x, x_new.y), 2, (0,0,0))
                cv2.line(img, (x_new.parent.x, x_new.parent.y), (x_new.x, x_new.y), (255,0,0))
            
                # Keep showing the image for a bit even
                # if we don't add a new node and edge
            cv2.imshow('image', img)
            cv2.waitKey(1)

        plan = self._follow_parent_pointers(x_best)
        dest_state.parent = x_best
        plan.append(dest_state)

        draw_plan(img, plan, [], "rrt_result.png", bgr=(0,0,255), thickness=2)
        draw_plan(img, plan, bgr=(0,0,255), thickness=2)
        cv2.waitKey(0)

        plt.plot(num_nodes_list, path_length_list, label='Optimal Path Length')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Path Length')
        plt.title('Path Length vs Number of Nodes in RRT')
        plt.legend()
        plt.savefig('rrt_opt_length_vs_num_nodes.png')
        plt.show()
        print(list(zip(num_nodes_list, path_length_list)))
        return [start_state]
    

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: rrt_planner.py occupancy_grid.pkl")
    #     sys.exit(1)


    max_num_steps = 1000     # max number of nodes to be added to the tree
    max_steering_radius = 50 # pixels
    dest_reached_radius = 50 # pixels


    # pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    # world = pickle.load(pkl_file)
    # pkl_file.close()
    # start_state = State(10, 10, None)
    # dest_state = State(10, 250, None)

    # world = cv2.imread('./worlds/complex_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)
        
    # world = cv2.imread('./worlds/complex_maze_concave.png')
    # start_state = State(170, 120, None)
    # dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/cluttered.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)


    # world = cv2.imread('./worlds/simple_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)
    # rrt = InformedRRTPlanner(world, start_state, dest_state)
    # plan = rrt.plan(start_state,
    #                 dest_state,
    #                 max_num_steps,
    #                 max_steering_radius,
    #                 dest_reached_radius)
    
    # world = cv2.imread('./worlds/complex_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)
    # rrt = InformedRRTPlanner(world, start_state, dest_state)
    # plan = rrt.plan(start_state,
    #                 dest_state,
    #                 max_num_steps,
    #                 max_steering_radius,
    #                 dest_reached_radius)
    
    # world = cv2.imread('./worlds/complex_maze_concave.png')
    # start_state = State(170, 120, None)
    # dest_state = State(1000, 650, None)
    # rrt = InformedRRTPlanner(world, start_state, dest_state)
    # plan = rrt.plan(start_state,
    #                 dest_state,
    #                 max_num_steps,
    #                 max_steering_radius,
    #                 dest_reached_radius)
    
    world = cv2.imread('./worlds/cluttered.png')
    start_state = State(40, 40, None)
    dest_state = State(500, 325, None)
        
    # world = cv2.imread('./worlds/cluttered.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)

    # start_state = State(10, 10, None)
    # dest_state = State(500, 500, None)
    # dest_state = State(215, 500, None)
    # dest_state = State(575, 70, None)

    rrt = InformedRRTPlanner(world, start_state, dest_state)
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)
    
    # world = cv2.imread('./worlds/floor_plan.png')
    # start_state = State(70, 860, None)
    # dest_state = State(1260, 100, None)

    # rrt = InformedRRTPlanner(world, start_state, dest_state)

    # plan = rrt.plan(start_state,
    #                 dest_state,
    #                 max_num_steps,
    #                 max_steering_radius,
    #                 dest_reached_radius)
    
    # dest_state = State(1250, 850, None)

    # rrt = InformedRRTPlanner(world, start_state, dest_state)

    # plan = rrt.plan(start_state,
    #                 dest_state,
    #                 max_num_steps,
    #                 max_steering_radius,
    #                 dest_reached_radius)


    # world = cv2.imread('./worlds/floor_plan_cleaned.png')
    # start_state = State(80, 820, None)
    # dest_state = State(1210, 90, None)
    # rrt = InformedRRTPlanner(world, start_state, dest_state)
    # plan = rrt.plan(start_state,
    #                 dest_state,
    #                 max_num_steps,
    #                 max_steering_radius,
    #                 dest_reached_radius)
    
    dest_state = State(1210, 820, None)
    rrt = InformedRRTPlanner(world, start_state, dest_state)
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)

    world = cv2.imread('./worlds/regular.png')
    start_state = State(30, 25, None)
    dest_state = State(925, 720, None)
    rrt = InformedRRTPlanner(world, start_state, dest_state)
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)

    world = cv2.imread('./worlds/irregular.png')
    start_state = State(40, 35, None)
    dest_state = State(800, 645, None)
    rrt = InformedRRTPlanner(world, start_state, dest_state)
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)

    world = cv2.imread('./worlds/narrow.png')
    start_state = State(35, 35, None)
    dest_state = State(1125, 900, None)
    rrt = InformedRRTPlanner(world, start_state, dest_state)
    
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)
