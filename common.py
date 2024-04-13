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

from plotting_utils import *
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
    

class RRTPlanner:
    """
    Base RRT Planner class
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

        self.delta = 5
    
    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-self.delta:state.y+self.delta, state.x-self.delta:state.x+self.delta] == 0).all()
    
    def sample_state(self):
        """
        Sample a new FREE state uniformly randomly on the image.
        """
        #TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
        rows, cols = self.world.shape[0], self.world.shape[1]
        state = None

        x = random.randint(0 + self.delta, cols - 1 - self.delta)
        y = random.randint(0 + self.delta, rows - 1 - self.delta)
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
        """
        Returns closest node to state from tree_nodes.
        """
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

        s_new = State(0, 0, s_nearest)

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
        
        # Ensure not out of bounds:
        rows, cols = self.world.shape[0], self.world.shape[1]
        if s_new.x < self.delta:
            s_new.x = self.delta
        elif s_new.x > cols - 1 - self.delta:
            s_new.x = cols - 1 - self.delta
        
        if s_new.y < self.delta:
            s_new.y = self.delta
        elif s_new.y > rows - 1 - self.delta:
            s_new.y = rows - 1 - self.delta

        return s_new
    
    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False

        max_checks = 100
        for i in range(max_checks):
            ratio = float(i) / max_checks

            x = int(s_from.x + ratio * (s_to.x - s_from.x))
            y = int(s_from.y + ratio * (s_to.y - s_from.y))

            inteprolated_state = State(x, y, s_from)

            if not (self.state_is_free(inteprolated_state)):
                return False

        # Otherwise the line is free, so return true
        return True
    
    def near(self, state, tree_nodes, radius):
        """
        Returns all the tree nodes within radius of state.
        """

        near_nodes = []

        for node in tree_nodes:
            if (node.euclidean_distance(state) < radius):
                near_nodes.append(node)
        
        return near_nodes
    
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
    
    def improved_hybrid_sample(self, dest_state, prev_sample, hybrid_lambda):

        hybrid_lambda_r = random.random()
        s_rand = self.sample_state()

        if (hybrid_lambda_r > hybrid_lambda):
            # Find a better sample than the previously sampled node
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

        # Find the furthest parent s_rand can directly connect to
        s_int = s_parent.parent
        while (s_int != None):

            if self.path_is_obstacle_free(s_int, s_rand):
                s_parent = s_int

            s_int = s_int.parent

        return s_parent
    
    def create_node(self, s_rand, s_parent, dichotomy):
        # Assumption: s_rand and s_parent.parent have an obstacle between them
        # Assumption: s_parent.parent != None

        s_allow = s_parent
        s_dic = None

        s_forbid = s_parent.parent
        while (s_allow.euclidean_distance(s_forbid) > dichotomy):
            # Problem encountered here infinite loop
            temp_x = int((s_allow.x + s_forbid.x) / 2)
            temp_y = int((s_allow.y + s_forbid.y) / 2)
            s_dic = State(temp_x, temp_y, None)

            # Assertion error is raised here sometimes
            if (self.path_is_obstacle_free(s_rand, s_dic)):
                s_allow = s_dic
            else:
                s_forbid = s_dic

        s_forbid = s_rand
        while (s_allow.euclidean_distance(s_forbid) > dichotomy):

            temp_x = int((s_allow.x + s_forbid.x) / 2)
            temp_y = int((s_allow.y + s_forbid.y) / 2)
            s_dic = State(temp_x, temp_y, None)

            if (self.path_is_obstacle_free(s_parent.parent, s_dic)):
                s_allow = s_dic
            else:
                s_forbid = s_dic

        s_parent = s_allow

        return s_parent
    
    def rewire(self, s_rand, closest_region):
        for s_near in closest_region:
            if self.path_is_obstacle_free(s_near, s_rand):
                if s_near.cost > s_rand.cost + s_near.euclidean_distance(
                        s_rand):
                    s_near.parent.children.remove(s_near)
                    s_near.cost = s_rand.cost + s_near.euclidean_distance(
                        s_rand)
                    s_rand.children.append(s_near)
                    s_near.parent = s_rand
    
    def improved_rewire(self, s_rand, closest_region):
        for s_near in closest_region:
            if self.path_is_obstacle_free(s_near, s_rand.parent): # Difference is here (s_rand.parent instead of s_rand)
                if s_near.cost > s_rand.parent.cost + s_near.euclidean_distance(
                        s_rand.parent):
                    s_near.parent.children.remove(s_near)
                    s_near.cost = s_rand.parent.cost + s_near.euclidean_distance(
                        s_rand.parent)
                    s_rand.parent.children.append(s_near)
                    s_near.parent = s_rand.parent
    
    # Abstract Method
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius, test=False, filename=None):

        pass
    
    # Abstract Method (different paramaters)
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, hybrid_lambda, test=False, filename=None):

        pass

    # Abstract Method (different paramaters)
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dichotomy, test=False, filename=None):

        pass
    
    # Abstract Method (different paramaters)
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, hybrid_lambda, dichotomy, test=False, filename=None):

        pass
