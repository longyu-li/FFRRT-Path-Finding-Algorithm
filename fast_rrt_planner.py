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

class FastRRTPlanner:
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
        return (self.occ_grid[state.y-5:state.y+5, state.x-5:state.x+5] == 0).all()


    def sample_state(self):
        """
        Sample a new FREE state uniformly randomly on the image.
        """
        #TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
        rows, cols = self.world.shape[0], self.world.shape[1]
        state = None

        x = random.randint(0, cols - 1)
        y = random.randint(0, rows - 1)
        state = State(x, y, None)

        # Ensures the state returned is free
        while (self.state_is_free(state) == False):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
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

        # max_checks = 100
        # for i in range(max_checks):
        #     ratio = float(i) / max_checks
        #     ...

        # ...or step distance, which I prefer

        step_distance = 5
        total_distance = s_from.euclidean_distance(s_to)
        num_steps = int(total_distance / step_distance)

        # s_from and s_to are basically next to each other
        # we assume s_from and s_to are in a free space, so return True
        if (num_steps == 0):
            return True

        for i in range(num_steps + 1):  # Include the last point (s_to)
            ratio = float(i) / num_steps

            x = int(s_from.x + ratio * (s_to.x - s_from.x))
            y = int(s_from.y + ratio * (s_to.y - s_from.y))

            inteprolated_state = State(x, y, s_from)

            if not (self.state_is_free(inteprolated_state)):
                return False

        # Otherwise the line is free, so return true
        return True

    ################################################################

    # These are the functions from the FF-RRT* paper. 
    # The previous functions were from Assignment 2 (with minor adjustments)

    def hybrid_sample(self, dest_state, prev_sample, hybrid_lambda):

        hybrid_lambda_r = random.random()
        s_rand = None

        if (hybrid_lambda_r < hybrid_lambda):
            s_rand = dest_state
        else:
            s_rand = self.sample_state()

            # Find a better sample than the previously sampled node
            # (see HybridSampling diagram from the FF-RRT* paper)
            while (abs(s_rand.x - dest_state.x) > abs(dest_state.x - prev_sample.x)) and \
                    (abs(s_rand.y - dest_state.y) > abs(dest_state.y - prev_sample.y)):
                
                s_rand = self.sample_state()


        return s_rand

    def improved_choose_parent(self, s_rand, near_nodes, s_nearest):

        c_min = 0
        s_parent = s_nearest

        for node in near_nodes:
            if self.path_is_obstacle_free(node, s_rand):
                c = node.cost + node.euclidean_distance(s_rand)
                if (c_min == 0 or c < c_min):
                    s_parent = node
                    c_min = c
        
        # Looks if s_rand can directly connect to one of
        # s_parent's parents, and picks the furthest parent
        s_parent = self.backtracking(s_rand, s_parent)

        return s_parent
    
    def backtracking(self, s_rand, s_parent):

        # Basically, find the furthest parent s_rand can directly connect to
        s_int = s_parent.parent
        while (s_int != None):

            if self.path_is_obstacle_free(s_int, s_rand):
                s_parent = s_int

            s_int = s_int.parent

        return s_parent

    def near(self, state, tree_nodes, radius):
        """
        Returns all the tree nodes within radius of state.
        """

        near_nodes = []

        for node in tree_nodes:
            if (node.euclidean_distance(state) < radius):
                near_nodes.append(node)
        
        return near_nodes
    
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
        prev_sample = start_state

        for step in range(max_num_steps):

            s_rand = self.hybrid_sample(dest_state, prev_sample, hybrid_lambda)
            s_nearest = self.find_closest_state(tree_nodes, s_rand)
            s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)

            near_nodes = self.near(s_new, tree_nodes, max_steering_radius)
            s_parent = self.improved_choose_parent(s_new, near_nodes, s_nearest)

            if self.path_is_obstacle_free(s_parent, s_new):

                # Update parent and cost here
                s_new.parent = s_parent
                s_new.cost += s_parent.cost + s_new.euclidean_distance(s_parent)

                tree_nodes.add(s_new)
                s_parent.children.append(s_new)

                # Calculate the length of the current optimal path at each step
                s_nearest_to_goal = self.find_closest_state(tree_nodes, dest_state)
                curr_opt_path_length = s_nearest_to_goal.euclidean_distance(dest_state)
                path_length_list.append(curr_opt_path_length)

                # Calculate the number of nodes in the tree at each step
                num_nodes = len(tree_nodes)
                num_nodes_list.append(num_nodes)

                if self.path_is_obstacle_free(s_new, dest_state):
                    dest_state.parent = s_new
                    plan = self._follow_parent_pointers(dest_state)
                    break

                # plot the new node and edge
                cv2.circle(img, (s_new.x, s_new.y), 2, (0,0,0))
                cv2.line(img, (s_parent.x, s_parent.y), (s_new.x, s_new.y), (255,0,0))
            
                prev_sample = s_new
        

            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            cv2.imshow('image', img)
            cv2.waitKey(10)

        # draw_plan(img, plan, [], "rrt_result.png", bgr=(0,0,255), thickness=2)
        draw_plan(img, plan, bgr=(0,0,255), thickness=2)
        cv2.waitKey(0)

        plt.plot(num_nodes_list, path_length_list, label='Optimal Path Length')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Path Length')
        plt.title('Path Length vs Number of Nodes in RRT')
        plt.legend()
        plt.savefig('rrt_opt_length_vs_num_nodes.png')
        plt.show()
        
        return [start_state]



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rrt_planner.py occupancy_grid.pkl")
        sys.exit(1)

    # pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    # world = pickle.load(pkl_file)
    # pkl_file.close()
        
    # world = cv2.imread('./worlds/simple_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/complex_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)
        
    # world = cv2.imread('./worlds/complex_maze_concave.png')
    # start_state = State(170, 120, None)
    # dest_state = State(1000, 650, None)

    world = cv2.imread('./worlds/cluttered.png')
    start_state = State(40, 40, None)
    dest_state = State(1000, 650, None)

    rrt = FastRRTPlanner(world, start_state, dest_state)

    max_num_steps = 20000     # max number of nodes to be added to the tree
    max_steering_radius = 50 # pixels
    dest_reached_radius = 50 # pixels
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)



