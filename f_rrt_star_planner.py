#!/usr/bin/python
import math
import sys
import time
import pickle
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from itertools import product
from math import cos, sin, pi, sqrt

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
        self.cost = 0

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
        return sqrt((state.x - self.x) ** 2 + (state.y - self.y) ** 2)


class FRrtStarPlanner:
    """
    Applies the RRT algorithm on a given grid world
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:, :, 0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y - 5:state.y + 5,
                state.x - 5:state.x + 5] == 0).all()

    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image.
        """
        # TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
        rows, cols = self.world.shape[:2]
        x = np.random.randint(0, cols)
        y = np.random.randint(0, rows)

        state = State(x, y, None)

        # Ensures the state returned is free
        while self.state_is_free(state) == False:
            x = random.randint(0, cols)
            y = random.randint(0, rows)
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

        if s_nearest.euclidean_distance(s_rand) <= max_radius:
            x = s_rand.x
            y = s_rand.y
        else:
            # 2-D vector from s_nearest to s_rand
            v = np.array([s_rand.x - s_nearest.x, s_rand.y - s_nearest.y])

            # Compute a scalar that will take the vector v's magnitude to
            # max_radius
            scalar = max_radius / s_nearest.euclidean_distance(s_rand)
            v = scalar * v

            rows, cols = self.world.shape[:2]

            # boundary check. Flip direction of vector v if the head of
            # v goes out of bounds.
            if int(v[0]) + s_nearest.x > cols - 1 or int(
                    v[1]) + s_nearest.y > rows - 1:
                v = -v

            # With s_nearest as the origin, compute the x and y values of the
            # point at the head of vector v
            x = int(v[0]) + s_nearest.x
            y = int(v[1]) + s_nearest.y

        s_new = State(x, y, s_nearest)
        assert (s_nearest.euclidean_distance(s_new) <= max_radius)

        return s_new

    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False

        # if the euclidean distance between s_from and s_to is zero, there
        # is no line path between them.
        if s_from.euclidean_distance(s_to) == 0:
            return False

        max_checks = 500
        for i in range(max_checks):
            # TODO: check if the inteprolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False

            # Used to interpolate state
            dist = float(i) / max_checks * s_from.euclidean_distance(s_to)

            # 2-D vector from s_to to s_from
            v = np.array([s_to.x - s_from.x, s_to.y - s_from.y])

            # To compute the interpolated state, take the interpolated dist
            # away from the s_to state and compute a scalar that takes
            # the magnitude of v to the interpolated dist
            scalar = dist / s_from.euclidean_distance(s_to)
            v = scalar * v

            # Establish interpolated state
            s_new = State(int(v[0]) + s_from.x, int(v[1]) + s_from.y, s_from)

            # Check if the interpolated state is not free
            if not (self.state_is_free(s_new)):
                return False

        # Otherwise the line is free, so return true
        return True

    #########################################################################

    def Near(self, tree_nodes, s_rand, r_near):
        """
        Returns the circular region centered around the sampled state s_rand
        with radius length r_near, that contains the states closest to
        s_rand in the current tree.
        """
        closest_region = []
        for node in tree_nodes:
            dist = node.euclidean_distance(s_rand)
            if dist < r_near:
                closest_region.append(node)

        return closest_region

    def CreatNode(self, s_rand, s_parent, dichotomy):
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

    def ChooseParent(self, s_rand, s_nearest, closest_region):
        c_min = 0
        s_parent = s_nearest

        for node in closest_region:
            if self.path_is_obstacle_free(node, s_rand):
                c = node.cost + node.euclidean_distance(s_rand)
                if (c_min == 0 or c < c_min):
                    s_parent = node
                    c_min = c

        # Looks if s_rand can directly connect to one of
        # s_parent's parents, and picks the furthest parent
        s_parent = self.Backtracking(s_rand, s_parent)

        return s_parent

    def Backtracking(self, s_rand, s_parent):
        # Basically, find the furthest parent s_rand can directly connect to
        s_int = s_parent.parent
        while (s_int != None):

            if self.path_is_obstacle_free(s_int, s_rand):
                s_parent = s_int

            s_int = s_int.parent

        return s_parent

    def Rewire(self, s_rand, closest_region):
        for s_near in closest_region:
            if self.path_is_obstacle_free(s_near, s_rand):
                if s_near.cost > s_rand.cost + s_near.euclidean_distance(
                        s_rand):
                    s_near.parent.children.remove(s_near)
                    s_near.cost = s_rand.cost + s_near.euclidean_distance(
                        s_rand)
                    s_rand.children.append(s_near)
                    s_near.parent = s_rand

    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius,
             dichotomy):
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

        # For plotting optimal path length as a function of tree nodes
        # path_lengths = [float("Inf")]
        # optimal_path_lengths = [float("Inf")]

        for step in range(max_num_steps):

            # TODO: Use the methods of this class as in the slides to
            # compute s_new
            # sample a random state uniformly, then determine the closest state
            # and steer it towards the new state.
            s_rand = self.sample_state()
            s_nearest = self.find_closest_state(tree_nodes, s_rand)
            s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)

            if not self.state_is_free(s_new):
                continue

            if self.path_is_obstacle_free(s_new, s_nearest):
                closest_region = self.Near(tree_nodes, s_new, max_steering_radius)
                s_parent = self.ChooseParent(s_new, s_nearest, closest_region)

                if s_parent != start_state:
                    s_anc = s_parent.parent
                    s_parent = self.CreatNode(s_new, s_parent, dichotomy)

                    s_parent.parent = s_anc
                    s_new.parent = s_parent

                    s_parent.cost = s_anc.cost + s_parent.euclidean_distance(s_anc)
                    s_new.cost = s_parent.cost + s_new.euclidean_distance(s_parent)

                    s_parent.children.append(s_new)
                    s_anc.children.append(s_parent)

                    tree_nodes.add(s_parent)
                    tree_nodes.add(s_new)

                    # plot the new parent
                    cv2.circle(img, (s_parent.x, s_parent.y), 2, (0, 0, 0))
                    cv2.line(img, (s_anc.x, s_anc.y), (s_parent.x, s_parent.y),
                             (255, 0, 0))

                else:
                    s_new.parent = s_parent
                    s_new.cost = s_parent.cost + s_new.euclidean_distance(s_parent)
                    s_parent.children.append(s_new)
                    tree_nodes.add(s_new)

                if self.path_is_obstacle_free(s_new, dest_state):
                    s_anc = s_new.parent
                    s_parent = self.CreatNode(dest_state, s_new, dichotomy)
                    dest_state.parent = s_parent
                    s_parent.parent = s_anc
                    plan = self._follow_parent_pointers(dest_state)

                    break

                    # compute the distance from the start state to the dest
                    # state, following the path denoted by plan to determine the
                    # actual length of the path.
                    # plan_length = 0
                    # for index in range(len(plan)-1):
                    #     plan_length += plan[index].euclidean_distance(plan[index + 1])
                    # path_lengths.append(plan_length)



                # keep track of optimal path for plot
                # optimal_path_lengths.append(min(path_lengths))

                self.Rewire(s_new, closest_region)

                cv2.circle(img, (s_new.x, s_new.y), 2, (0, 0, 0))
                cv2.line(img, (s_parent.x, s_parent.y), (s_new.x, s_new.y),
                         (255, 0, 0))

            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            cv2.imshow('image', img)
            cv2.waitKey(10)

        # plt.plot(list(range(len(tree_nodes))), optimal_path_lengths)
        # plt.xlabel("Number of nodes in tree")
        # plt.ylabel("Length of the optimal path (in pixels)")

        # plt.title("Optimal length vs. Number of nodes")
        # plt.show()

        draw_plan(img, plan, bgr=(0, 0, 255), thickness=2)
        cv2.waitKey(0)
        return [start_state]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: f_rrt_star_planner.py occupancy_grid.pkl")
        sys.exit(1)

    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = cv2.imread(sys.argv[1])

    # world = cv2.imread('./worlds/simple_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/complex_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/complex_maze_concave.png')
    # start_state = State(170, 120, None)
    # dest_state = State(1000, 650, None)


    f_rrt_star = FRrtStarPlanner(world)

    start_state = State(40, 40, None)
    dest_state = State(1000, 650, None)

    max_num_steps = 3000  # max number of nodes to be added to the tree
    max_steering_radius = 30  # pixels
    dest_reached_radius = 50  # pixels
    dichotomy = 2
    plan = f_rrt_star.plan(start_state,
                           dest_state,
                           max_num_steps,
                           max_steering_radius,
                           dichotomy)
