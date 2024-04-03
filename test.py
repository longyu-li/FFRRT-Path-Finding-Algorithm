import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from fast_rrt_planner import FastRRTPlanner
from ff_rrt_star_planner import FFRRTPlanner
from informed_rrt_star_planner_v2 import InformedRRTPlanner, State
from f_rrt_star_planner import FRRTPlanner


def create_plot(algo_data, indicator_label):
    """
    Creates a box plot displaying the distribution of performance indicators for
    the Informed-RRT*, Fast-RRT*, F-RRT*, and FF-RRT* path planning algorithms.

    algo_data is expected to be an array containing performance metric data
    corresponding to each of the four algorithms, Informed-RRT*, Fast-RRT*,
    F-RRT*, and FF-RRT*. Explicitly, it has the following form:
        algo_data = [Informed-RRT* data, Fast-RRT* data, F-RRT* data, FF-RRT* data]
        *Note: data for each algorithm should be in this order

    Each of Informed-RRT* data, Fast-RRT* data, F-RRT* data, and FF-RRT* data are
    arrays. The performance evaluation of the paper ran the simulations about
    100 times, so the length of the arrays should be 100 (subject to change).
    Additionally, `data` corresponds to one of the following indicators:
        C^{i}: cost of the generated path of algorithm i
        N^{i}: number of iterations to generate the path by algorithm i
        T^{i}: computation time of algorithm i
        ...
        *Note: These are the indicators mentioned in the paper

    indicator_label should be a string
    """
    algos = ["Informed-RRT*", "Fast-RRT*", "F-RRT*", "FF-RRT*"]
    plt.boxplot(algo_data, labels=algos)
    plt.ylabel(indicator_label)
    plt.show()


if __name__ == "__main__":
    # Takes in the world image as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: test.py path_to_img")
        sys.exit(1)

    world = cv2.imread(sys.argv[1])

    # Initialize start and dest states
    start_state = State(40, 40, None)
    dest_state = State(1000, 650, None)

    # Initialize mapping algorithms
    informed_rrt_star = InformedRRTPlanner(world, start_state, dest_state)
    fast_rrt_star = FastRRTPlanner(world, start_state, dest_state)
    f_rrt_star = FRRTPlanner(world)
    ff_rrt_star = FFRRTPlanner(world, start_state, dest_state)

    # Initialize other planning parameters
    max_num_steps = 100000  # max number of nodes to be added to the tree
    max_steering_radius = 50  # pixels
    dest_reached_radius = 50  # pixels
    dichotomy = 2

    # instantiate the arrays to store algorithm cost, iterations,
    # and computation time for each algorithm
    algos_c = [[], [], [], []]
    algos_n = [[], [], [], []]
    algos_t = [[], [], [], []]

    # runs 100 simulations (not sure if we have the time to run this many
    # simulations) and stores path cost, number of iterations to get path,
    # and computation time in respective arrays
    for sim_num in range(1, 101):
        # Assumes that the plan function returns the cost, and number of
        # iterations to find the optimal path

        start_time = time.perf_counter()
        irrt_path, irrt_cost, irrt_nodes = informed_rrt_star.plan()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        algos_t[0].append(execution_time)

        start_time = time.perf_counter()
        fast_path, fast_cost, fast_nodes = fast_rrt_star.plan()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        algos_t[1].append(execution_time)

        start_time = time.perf_counter()
        frrt_path, frrt_cost, frrt_nodes = f_rrt_star.plan()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        algos_t[2].append(execution_time)

        start_time = time.perf_counter()
        ffrrt_path, ffrrt_cost, ffrrt_nodes = ff_rrt_star.plan()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        algos_t[3].append(execution_time)

        algos_c[0].append(irrt_cost)
        algos_c[1].append(fast_cost)
        algos_c[2].append(frrt_cost)
        algos_c[3].append(ffrrt_cost)

        algos_n[0].append(irrt_nodes)
        algos_n[1].append(fast_nodes)
        algos_n[2].append(frrt_nodes)
        algos_n[3].append(ffrrt_nodes)
        

    # Create the box plots displaying the performance indicators
    create_plot(algos_c, "C^{i}")
    create_plot(algos_n, "N^{i}")
    create_plot(algos_t, "T^{i}")
