import random
import math
from statistics import mean
import timeit
import csv
import sys


class GraphMatrix(object):
    """Graph object for faster calculation"""

    def __init__(self, numpoints, dimension):
        self.numpoints = numpoints
        self.matrix = []

        # Create matrix
        if dimension == 0:
            self.matrix = [[0] * numpoints for x in range(numpoints)]

            for i in range(1, numpoints):
                for j in range(0, i):
                    # Assign weights randomly when dimension is 0
                    self.matrix[i][j] = random.random()

        # Assumes dimensions are 0, 2, 3, or 4
        if dimension != 0:

            # 1 dimensional array of points:
            points = [[0] * dimension for x in range(numpoints)]
            for r in range(0, numpoints):
                for c in range(dimension):
                    points[r][c] = random.random()

            # Assign matrix with point distances:
            self.matrix = [[0] * numpoints for x in range(numpoints)]

            # Fill in distances:
            for row in range(1, numpoints):
                for col in range(0, row):
                    point1 = points[row]
                    point2 = points[col]

                    # Calculate Euclidean Distance between 2 points here
                    sumSquares = [math.pow(point1[i] - point2[i], 2)
                    for i in range(len(point1))]
                    distance = math.sqrt(sum(sumSquares))
                    self.matrix[row][col] = distance

        # initialize one random vertex as root and prepare tracking lists
        self.root_index = random.randint(0, numpoints - 1)
        self.all_weights = self.get_weights(self.root_index)
        self.start = [None] * numpoints
        self.start[self.root_index] = 0
        self.min_index = self.root_index
        self.min_value = 0

    # Get list of weights for all of vertex edges / return weight of 0 at index i
    def get_weights(self, i):
        weights = [0] * self.numpoints

        for j in range(i):
            weights[j] = self.matrix[i][j]

        for j in range(i + 1, self.numpoints):
            weights[j] = self.matrix[j][i]

        return weights

    # Check for weight is 0 or not
    def mst_weight(self):
        s = 0

        for weight in self.start:
            if weight:
                s += weight
        return s
    
    def get_max_weights(self):
        max_weight = max(self.start)
        return max_weight

    # Set and calculate weights/distance for each graph
    def prim_mst(self):
        """Complete's Prim's algorithm and return the final mst"""
        # Source: https://www.programiz.com/dsa/prim-algorithm
        start = timeit.default_timer()

        for _ in range(self.numpoints):
            weights = self.get_weights(self.min_index)
            self.start[self.min_index] = self.min_value

            # reset minimum value
            self.min_value = float("inf")
            for i, weight in enumerate(weights):
                if self.start[i] is None:

                    # Compare weights for each record based on all weights
                    # and min weights
                    if weight < self.all_weights[i]:
                        self.all_weights[i] = weight

                    if self.all_weights[i] < self.min_value:
                        self.min_value = self.all_weights[i]
                        self.min_index = i

        stop = timeit.default_timer()

        # Return total weight of the mst and average mst weight
        weight = self.mst_weight()
        # average_weight = sum(self.start) / len(self.start)
        mst_max_weight = self.get_max_weights()
        time = stop - start
        return weight, time, mst_max_weight # average_weight


# Primary driver of running different number of graph points, trials, 
# and dimensions
def run_mst(numtrials, numpoints, dimension, csvwriter):
    # Set seed for number of trials
#    random.seed(113)
    
    weights = []
    tot_time = []
    max_weight = []
    for r in range(numtrials):
        graph = GraphMatrix(numpoints, dimension)
        mst_weight, time, max_mst_weight = graph.prim_mst()
        weights.append(mst_weight)
        tot_time.append(time)
        max_weight.append(max_mst_weight)
    average_weight = mean(weights)
    exec_time = mean(tot_time) # average runtime for 1 trial.
    maximum_mst_weight = mean(max_weight)
        # This print does not show mst weights
    print(
        "Average MST Weight: {0}, NumPoints: {1},\
 NumTrials {2}, Dimension {3}, Execution Time {4}, max MST Weight {5}".format(
            average_weight, numpoints, numtrials, dimension, exec_time,
            maximum_mst_weight))

    csvwriter.writerow({"NumPoints": numpoints, "Dimension": dimension,
                        "Average Weight": average_weight,
                        "NumTrials": numtrials, 
                        'Execution Time': exec_time,
                        'Maximum MST Weight': maximum_mst_weight})


if __name__ == "__main__":
    # Assumes executions is done with the cmd:
    # python randmst.py numpoints numtrials dimension
    numpoints = int(sys.argv[1])
    numtrials = int(sys.argv[2])
    dimension = int(sys.argv[3])

    with open(('./mst_dimension' + str(dimension) + '_points' + str(numpoints)
                + '.csv'), 'w') as csvfile:
        fieldnames = ["NumPoints", "Dimension",
                      "Average Weight", "NumTrials", 'Execution Time',
                      'Maximum MST Weight']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        run_mst(numtrials, numpoints, dimension, csvwriter)
        print("")
        print("Execution has completed and csv file made\
 with the above results in the current working directory.")

    # Loop through possible parameters for quick analysis or testing
    # with open('./mst_sizes_testing.csv', 'a') as csvfile:
    #    fieldnames = ["NumPoints", "Dimension", "Size", "Execution Time", "Average Weight"]
    #    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # set dimensions for 0, 2, 3, and 4
    # for dimension in [0, 2, 3, 4]:

    # set number of points in graph to the 2^n
    #    for point_power in range(8, 12):
    #        run_mst(5, int(math.pow(2, point_power)), dimension, csvwriter)
