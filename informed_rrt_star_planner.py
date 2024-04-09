#!/usr/bin/python
from common import *

class InformedRRTPlanner(RRTPlanner):
    """
    Implementation of the Informed RRT* Planning algorithm.
    """

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
    
    def SampleFreeSpace(self, dest_state, hybrid_lambda):
    
        hybrid_lambda_r = random.random()
        s_rand = None

        if (hybrid_lambda_r < hybrid_lambda):
            s_rand = dest_state
        else:
            s_rand = self.sample_state()

        return s_rand
    
    def Sample(self, c_max, c_min, x_center, C, dest_state, hybrid_lambda_r):
        if c_max < np.inf:
            r = [c_max / 2.0,
                 sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            L = np.diag(r)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand_temp = np.dot(np.dot(C, L), x_ball)
                x_rand = State(int(x_rand_temp[(0, 0)] + x_center.x), int(x_rand_temp[(1, 0)] + x_center.y), None)
                if self.state_is_free(x_rand):
                    break
        else:
            x_rand = self.SampleFreeSpace(dest_state, hybrid_lambda_r)
        
        return x_rand
    
    def InGoalRegion(self, node, dest_state, dest_reached_radius):
        if node.euclidean_distance(dest_state) < dest_reached_radius:
            return True

        return False

    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius, test, filename):
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

        # output format = [[curr_best_cost], [num_nodes], [time]]
        output = [[], [], []]
        start_time = time.perf_counter()

        X_soln = set()
        for step in range(max_num_steps):
            num_nodes_list.append(num_nodes_list[step-1] + 1 if len(num_nodes_list) > 0 else 1)
            c_best = float('inf')
            for node in X_soln:
                if node.cost < c_best:
                    # print("num nodes", "path cost")
                    # print(num_nodes_list[step], node.cost)
                    c_best = node.cost
                    x_best = node

            path_length_list.append(c_best)

            # Update output every 100 nodes
            if (test == True) and \
                (c_best != float('inf')) and \
                (num_nodes_list[step] % 100 == 0 or output[0] == []): 

                curr_time = time.perf_counter() - start_time
                output[0].append(c_best + x_best.euclidean_distance(dest_state))
                output[1].append(num_nodes_list[step])
                output[2].append(curr_time)

            
            if (test == True) and (step % 1000 == 0):
                print(step)

            x_rand = self.Sample(c_best, dist, x_center, C, dest_state, hybrid_lambda)
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
            if (test == False):
                cv2.imshow('image', img)
                cv2.waitKey(10)

        if (x_best == start_state):
            print("Informed-RRT: No path found!")
            if (test == False):
                return (None, -1, -1)
            else:
                return None, -1
        
        plan = self._follow_parent_pointers(x_best)
        dest_state.parent = x_best
        plan.append(dest_state)

        if (filename != None):
            draw_plan_and_save(img, plan, [], filename, bgr=(0,0,255), thickness=2)

        if (test == False):
            draw_plan(img, plan, bgr=(0,0,255), thickness=2)
            cv2.waitKey(0)

        # plt.plot(num_nodes_list, path_length_list, label='Optimal Path Length')
        # plt.xlabel('Number of Nodes')
        # plt.ylabel('Path Length')
        # plt.title('Path Length vs Number of Nodes in RRT')
        # plt.legend()
        # plt.savefig('rrt_opt_length_vs_num_nodes.png')
        # plt.show()
        
        # Calculaute optimal path cost
        cost_of_optimal_path = 0
        curr_node = dest_state
        while curr_node.parent != None:
            cost_of_optimal_path += curr_node.euclidean_distance(curr_node.parent)
            curr_node = curr_node.parent
        
        if (test == False):
            return (plan, cost_of_optimal_path, len(tree_nodes))
        else:
            return output, cost_of_optimal_path
    

if __name__ == "__main__":
        
    world = cv2.imread('./worlds/simple_maze.png')
    start_state = State(40, 40, None)
    dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/complex_maze.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)
        
    # world = cv2.imread('./worlds/complex_maze_concave.png')
    # start_state = State(160, 70, None)
    # dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/cluttered.png')
    # start_state = State(40, 40, None)
    # dest_state = State(1000, 650, None)

    # world = cv2.imread('./worlds/floor_plan.png')
    # start_state = State(70, 860, None)
    # dest_state = State(1260, 100, None)
    # dest_state = State(1250, 850, None)

    # world = cv2.imread('./worlds/floor_plan_cleaned.png')
    # start_state = State(80, 820, None)
    # dest_state = State(1210, 90, None)
    # dest_state = State(1070, 800, None)

    # world = cv2.imread('./worlds/regular.png')
    # start_state = State(30, 25, None)
    # dest_state = State(925, 720, None)

    # world = cv2.imread('./worlds/irregular.png')
    # start_state = State(40, 35, None)
    # dest_state = State(800, 645, None)

    # world = cv2.imread('./worlds/narrow.png')
    # start_state = State(35, 35, None)
    # dest_state = State(1125, 900, None)

    informed_rrt_star = InformedRRTPlanner(world)
    
    max_num_steps = 2500     # max number of iterations
    max_steering_radius = 70  # pixels
    dest_reached_radius = 50  # pixels
    (plan, cost, num_nodes) = informed_rrt_star.plan(start_state,
                                                    dest_state,
                                                    max_num_steps,
                                                    max_steering_radius,
                                                    dest_reached_radius,
                                                    test=False,
                                                    filename="informed_rrt_result.png")
