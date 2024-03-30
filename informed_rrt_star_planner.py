#!/usr/bin/python
import sys
import time
import pickle
import numpy as np
import random
import cv2

from itertools import product
from math import cos, sin, pi, sqrt, log
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

class RRTStarPlanner:
    """
    Applies the RRT algorithm on a given grid world
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-5:state.y+5, state.x-5:state.x+5] == 0).all()


    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image.
        """
        #TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
        x = random.randint(0, len(self.world[0]) - 1)
        y = random.randint(0, len(self.world) - 1)
        return State(x, y, None)


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

    def steer_towards(self, s_nearest:State, s_rand:State, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:

        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y

        Otherwise, s_rand is farther than max_radius from s_nearest.
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.

        """
        x = 0
        y = 0
        # line from s_nearest to s_rand is these ^ coords
        distance = s_nearest.euclidean_distance(s_rand)
        if distance <= max_radius:
            x = s_rand.x
            y = s_rand.y
        else:
            #s_rand is x_1, s_nearest is x_0
            if s_rand.x == s_nearest.x:
                x = s_rand.x
                if s_rand.y < s_nearest.y:
                    y = s_nearest.y + max_radius
                else:
                    y = s_nearest.y -max_radius
            else:
                slope = (s_rand.y - s_nearest.y)/(s_rand.x - s_nearest.x)
                if s_nearest.x < s_rand.x:
                    x = s_nearest.x + max_radius/sqrt(1 + slope**2)
                else:
                    x = s_nearest.x - max_radius/sqrt(1 + slope**2)
                y = slope * (x - s_nearest.x) + s_nearest.y
            
            
        #TODO: populate x and y properly according to the description above.
        #Note: x and y are integers and they should be in {0, ..., cols -1}
        # and {0, ..., rows -1} respectively

        s_new = State(int(x), int(y), s_nearest)
        return s_new


    def path_is_obstacle_free(self, s_from:State, s_to:State):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False
        
        max_checks = 10
        distance = s_from.euclidean_distance(s_to)
        for i in range(max_checks):
            # TODO: check if the inteprolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False
            if not self.state_is_free(self.steer_towards(s_from, s_to, distance * (float(i)/ max_checks))):
                return False

        # Otherwise the line is free, so return true
        return True


    def near(self, tree_nodes, state, max_distance):
        close_states = []
        for node in tree_nodes:
            if node == state:
                continue
            dist = node.euclidean_distance(state)
            if dist < max_distance:
                close_states.append(state)

        return close_states
    
    def cost(self, state:State):
        parents = self._follow_parent_pointers(state)
        return len(parents)


    def sample_from_2d_unit_ball(self):
        length = np.sqrt(np.random.uniform(0, 1))
        angle = np.pi * np.random.uniform(0, 2)
        return np.array([[length * np.cos(angle)], [length * np.sin(angle)]])


    def rotation_to_world_frame(self, start_state, goal_state, c_min):
        a1 = np.array([[(goal_state.x - goal_state.y) / c_min],
                       [goal_state.y - start_state.y]])
        ones = np.array([[1, 0]])
        M = a1 @ ones
        U, Sigma, V_T = np.linalg.svd(M, True)
        return U @ np.diag([1, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
    
    def informed_sample(self, x_start:State, x_goal:State, c_max, C):
        if c_max < float('inf'):
            c_min = (x_goal.euclidean_distance(x_start))
            x_centre = [(x_start.x + x_goal.x)/2, (x_start.y + x_goal.y)/2]
            #for 3d change C and r to be 3 dimensional, maybe
            r_n = sqrt(c_max**2 - c_min**2)/2
            r = [c_max / 2, r_n]
            L = np.diag(r)
            while True:
                x_ball = self.sample_from_2d_unit_ball()
                _x_pre_rand = np.matmul(np.matmul(C, L), x_ball)
                x_rand = State(_x_pre_rand[0,0] + x_centre[0], _x_pre_rand[1,0] + x_centre[1], None)
                if(0 <= x_rand.x <= len(self.world[0]) - 1 and
                   0 <= x_rand.y <= len(self.world) -1):
                    break
        else:
            x_rand = self.sample_state()
        
        return x_rand
    
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

        X_soln = {}
        # image to be used to display the tree
        img = np.copy(self.world)

        plan = [start_state]
        plot_points = {1: float('inf')}
        C = self.rotation_to_world_frame(start_state, dest_state, start_state.euclidean_distance(dest_state))
        for step in range(max_num_steps):
            c_best = float('inf')
            for key in X_soln.keys():
                if X_soln[key] < c_best:
                    c_best = X_soln[key]

            #x_rand = self.sample_state()
            x_rand = self.informed_sample(start_state, dest_state, c_best, C)
            x_nearest = self.find_closest_state(tree_nodes, x_rand)
            x_new = self.steer_towards(x_nearest, x_rand, max_steering_radius)
            if self.path_is_obstacle_free(x_nearest, x_new):
                tree_nodes.add(x_new)

                X_near = self.near(tree_nodes, x_new, max_steering_radius)
                x_min = x_nearest
                c_min = self.cost(x_new)
                for x_near in X_near:
                    c_new = self.cost(x_near) + x_near.euclidean_distance(x_new)
                    if c_new < c_min:
                        if(self.path_is_obstacle_free(x_near, x_new)):
                            x_min = x_near 
                            c_min = c_new 

                x_min.children.append(x_new)
                x_new.parent = x_min 
                for x_near in X_near:
                    c_near = len(self._follow_parent_pointers(x_near))
                    c_new = len(self._follow_parent_pointers(x_new)) + x_new.euclidean_distance(x_near)
                    if c_new < c_near:
                        if self.path_is_obstacle_free(x_near, x_new):

                            x_parent = x_near.parent
                            x_parent.children.remove(x_near)
                            x_new.children.append(x_near)
                            x_near.parent = x_new
                
                plot_points[len(tree_nodes)] = plot_points[len(tree_nodes) - 1]

                if x_new.euclidean_distance(dest_state) < dest_reached_radius:
                    X_soln[x_new] = self.cost(x_new) * max_steering_radius
                    if self.cost(x_new) * max_steering_radius < plot_points[len(tree_nodes)]:
                        dest_state.parent = x_new
                        plot_points[len(tree_nodes)] = self.cost(dest_state) * max_steering_radius
                # plot the new node and edge
        for node in tree_nodes:
            cv2.circle(img, (node.x, node.y), 2, (0,0,0))
            for child in node.children:
                cv2.line(img, (child.x, child.y), (node.x, node.y), (255,0,0))


        # Keep showing the image for a bit even
        # if we don't add a new node and edge
        cv2.imshow('image', img)
        cv2.waitKey(10)
        x = []
        y = []
        for pair in list(plot_points.items()):
            if(pair[1] != -1):
                x.append(pair[0])
                y.append(pair[1])
        

        minx= float('infinity')
        for key in X_soln.keys():
            if X_soln[key] < minx:
                plan = self._follow_parent_pointers(key)
                minx = X_soln[key]

        draw_plan(img, plan, bgr=(0,0,255), thickness=2)
        plt.plot(x, y)
        plt.show()
        cv2.waitKey(0)

        return [start_state]



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rrt_planner.py occupancy_grid.pkl")
        sys.exit(1)

    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    rrt = RRTStarPlanner(world)

    start_state = State(10, 10, None)
    dest_state = State(500, 250, None)

    max_num_steps = 2000     # max number of nodes to be added to the tree
    max_steering_radius = 30 # pixels
    dest_reached_radius = 50 # pixels
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)

