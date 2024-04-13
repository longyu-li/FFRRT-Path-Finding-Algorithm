import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys
import time

from common import *
from informed_rrt_star_planner import InformedRRTPlanner


def create_plot(costs, nodes, times, filename):

    plt.figure(figsize=(8, 6))
    plt.plot(times, costs)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.xlabel("Time")
    plt.ylabel("Cost of Current Best Path")
    plt.savefig("irrt_costs_" + filename)
    # plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(times, nodes)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.xlabel("Time")
    plt.ylabel("Number of Nodes Visited")
    plt.savefig("irrt_nodes_" + filename)
    # plt.show()
    plt.close()


if __name__ == "__main__":

    environments = ["simple_maze.png", "complex_maze.png", "complex_maze_concave.png", 
                    "cluttered.png", "floor_plan_cleaned_1.png", "floor_plan_cleaned_2.png", 
                    "regular.png", "irregular.png", "narrow.png", "sauga_map.png"]

    
    with open('rrt_costs.txt', 'w') as file:

        for e in environments:

            print(e)

            world = cv2.imread("./worlds/" + e)

            # Initialize mapping algorithms
            informed_rrt_star = InformedRRTPlanner(world)

            # Initialize other planning parameters
            max_num_steps = 10000  # max number of nodes to be added to the tree
            max_steering_radius = 70  # pixels
            dest_reached_radius = 50  # pixels

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
            elif (e == "sauga_map.png"):
                start_state = State(295, 425, None)
                dest_state = State(4650, 4650, None)
                max_num_steps = 20000
                max_steering_radius = 150 
                dest_reached_radius = 150 
            else:
                print("Error finding environment!")

            print("Starting search...")

            output, final_cost = informed_rrt_star.plan(start_state,
                                                        dest_state,
                                                        max_num_steps,
                                                        max_steering_radius,
                                                        dest_reached_radius,
                                                        test=True,
                                                        filename="informed_rrt_" + e)
            
            if output != None:
                irrt_costs, irrt_nodes, irrt_time = output
                create_plot(irrt_costs, irrt_nodes, irrt_time, e)
            
            file.write("Cost of " + e + ": " + str(final_cost) + "\n")


        
