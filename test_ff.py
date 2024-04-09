import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys
import time

from common import *
from fast_rrt_star_planner import FastRRTPlanner
from ff_rrt_star_planner import FFRRTPlanner
from f_rrt_star_planner import FRRTPlanner


def create_plot(algo_data, indicator_label, filename):

    algos = ["Fast-RRT*", "F-RRT*", "FF-RRT*"]
    plt.figure(figsize=(8, 6))
    plt.boxplot(algo_data, labels=algos)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.ylabel(indicator_label)
    plt.savefig(filename)
    # plt.show()


if __name__ == "__main__":

    environments = ["simple_maze.png", "complex_maze.png", "complex_maze_concave.png", 
                    "cluttered.png", "floor_plan_cleaned_1.png", "floor_plan_cleaned_2.png", 
                    "regular.png", "irregular.png", "narrow.png"]
    
    for e in environments:

        print(e)

        world = cv2.imread("./worlds/" + e)

        start_state = dest_state = None

        if (e == "simple_maze.png"):
            start_state = State(40, 40, None)
            dest_state = State(1000, 650, None)

        elif (e == "complex_maze.png"):
            start_state = State(40, 40, None)
            dest_state = State(1000, 650, None)

        elif (e == "complex_maze_concave.png"):
            start_state = State(160, 70, None)
            dest_state = State(1000, 650, None)

        elif (e == "cluttered.png"):
            start_state = State(40, 40, None)
            dest_state = State(1000, 650, None)

        elif (e == "floor_plan_cleaned_1.png"):
            start_state = State(80, 820, None)
            dest_state = State(1210, 90, None)

        elif (e == "floor_plan_cleaned_2.png"):
            start_state = State(80, 820, None)
            dest_state = State(1070, 800, None)
        
        elif (e == "regular.png"):
            start_state = State(30, 25, None)
            dest_state = State(925, 720, None)
        
        elif (e == "irregular.png"):
            start_state = State(40, 35, None)
            dest_state = State(800, 645, None)
        
        elif (e == "narrow.png"):
            start_state = State(35, 35, None)
            dest_state = State(1125, 900, None)


        # Initialize mapping algorithms
        fast_rrt_star = FastRRTPlanner(world)
        f_rrt_star = FRRTPlanner(world)
        ff_rrt_star = FFRRTPlanner(world)

        # Initialize other planning parameters
        max_num_steps = 30000  # max number of nodes to be added to the tree
        max_steering_radius = 70  # pixels
        dest_reached_radius = 50  # pixels
        dichotomy = 2

        # instantiate the arrays to store algorithm cost, iterations,
        # and computation time for each algorithm
        algos_c = [[], [], []]
        algos_n = [[], [], []]
        algos_t = [[], [], []]

        # runs 30 simulations:
        for sim_num in range(1, 31):

            print(sim_num)

            start_time = time.perf_counter()
            fast_path, fast_cost, fast_nodes = fast_rrt_star.plan(start_state,
                                                                    dest_state,
                                                                    max_num_steps,
                                                                    max_steering_radius,
                                                                    dest_reached_radius,
                                                                    test=True, filename=None)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            if fast_path != None:
                algos_t[0].append(execution_time)

            start_time = time.perf_counter()
            frrt_path, frrt_cost, frrt_nodes = f_rrt_star.plan(start_state,
                                                                dest_state,
                                                                max_num_steps,
                                                                max_steering_radius,
                                                                dichotomy,
                                                                test=True, filename=None)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            if frrt_path != None:
                algos_t[1].append(execution_time)

            start_time = time.perf_counter()
            ffrrt_path, ffrrt_cost, ffrrt_nodes = ff_rrt_star.plan(start_state,
                                                                    dest_state,
                                                                    max_num_steps,
                                                                    max_steering_radius,
                                                                    dichotomy,
                                                                    test=True, filename=None)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            if ffrrt_path != None:
                algos_t[2].append(execution_time)

            if fast_path != None:
                algos_c[0].append(fast_cost)
            if frrt_path != None:
                algos_c[1].append(frrt_cost)
            if ffrrt_path != None:
                algos_c[2].append(ffrrt_cost)

            if fast_path != None:
                algos_n[0].append(fast_nodes)
            if frrt_path != None:
                algos_n[1].append(frrt_nodes)
            if ffrrt_path != None:
                algos_n[2].append(ffrrt_nodes)
            

        # Create the box plots displaying the performance indicators
        create_plot(algos_c, "Cost of Optimal Path", "cost_" + e)
        create_plot(algos_n, "Number of Nodes Visited", "num_nodes_" + e)
        create_plot(algos_t, "Time", "time_" + e)
