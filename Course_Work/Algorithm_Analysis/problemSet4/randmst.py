import random
import math
import timeit
import csv
import sys


class GraphMatrix(object):
    """Class Matrix object for easier manipulation"""

    def __init__(self, numpoints, dimension):
        self.numpoints = numpoints # number of vertices
        self.matrix = []

        # first create a matrix of distances
        if dimension == 0:
            self.matrix = [[0] * numpoints for x in range(numpoints)]
            # because we're filling in left-diagonal matrix, 
            # only one col is filled on first row,
            # then two on the second, cols just needs to go up to rows each loop.

            for row in range(1, numpoints):
                for col in range(0, row):
                    # assign weights randomly when dimension is 0
                    self.matrix[row][col] = random.random()

        # assumes dimensions are 0, 2, 3, or 4
        if dimension != 0:

            # 1D array of coords
            # Nested list contains the dimensions of a point.  
            coords = [[0] * dimension for x in range(numpoints)]
            for i in range(0, numpoints):
                for d in range(dimension):
                    coords[i][d] = random.random()

            # matrix of distances
            self.matrix = [[0] * numpoints for x in range(numpoints)]

            # Fill in distances:
            # This method of filling distance will leave some edges
            # with length 0, which means no edge exist.  
            for row in range(1, numpoints):
                for col in range(0, row):
                    point1 = coords[row]
                    point2 = coords[col]

                    # Calculate Euclidean Distance between 2 points here
                    sumSquares = [math.pow(point1[i] - point2[i], 2) 
                                    for i in range(len(point1))]
                    distance = math.sqrt(sum(sumSquares))
                    self.matrix[row][col] = distance

        # initialize one random vertex as root and prepare tracking lists
        self.root_index = random.randint(0, numpoints - 1)
        self.all_weights = self.get_weights(self.root_index)
        self.visited = [None] * numpoints
        self.visited[self.root_index] = 0
        self.min_index = self.root_index
        self.min_value = 0

    def get_weights(self, i):
        """Returns a list of the weights for vertex i's edges.
        Note that this returns a weight of 0 at index i."""
        weights = [0] * self.n

        for j in range(i):
            weights[j] = self.matrix[i][j]

        for j in range(i + 1, self.n):
            weights[j] = self.matrix[j][i]

        return weights


    def mst_weight(self):
        s = 0

        for weight in self.visited:
            if weight:  # check that it's not None (this is a temporary fix)
                s += weight

        return s

    def get_max_weights(self):
        max_weight = max(self.visited)
        return max_weight

    def prim_mst(self):
        """Complete's Prim's algorithm and return the final mst"""
        # Source: https://www.programiz.com/dsa/prim-algorithm
        start = timeit.default_timer()
        for _ in range(self.n):
            weights = self.get_weights(self.min_index)
            self.visited[self.min_index] = self.min_value

            # reset min value
            self.min_value = float("inf")
            for i, weight in enumerate(weights):
                if self.visited[i] is None:

                    if weight < self.all_weights[i]:
                        self.all_weights[i] = weight

                    if self.all_weights[i] < self.min_value:
                        self.min_value = self.all_weights[i]
                        self.min_index = i

        stop = timeit.default_timer()

        # Return total weight of the mst and average mst weight
        weight = self.mst_weight()
        average_weight = sum(self.visited) / len(self.visited)

        time = stop - start
        return weight, time, average_weight


# Primary driver of running different number of graph points, trials, 
# and dimensions
def run_mst(numtrials, numpoints, dimension, csvwriter):
    for r in range(numtrials):
        G = GraphMatrix(numpoints, dimension)

        mst_weight, time, average_weight = G.prim_mst()

        # This print does not state mst_weight
        print(
            "Average Weight: {0}, NumPoints: {1}, NumTrials {2}, Dimension\
            {3}, Execution Time {4}".format(
                average_weight, numpoints, numtrials, dimension, time
            ))

        csvwriter.writerow({"NumPoints": numpoints, "Dimension": dimension,
                            "Size": mst_weight, "Execution Time": time,
                            "Average Weight": average_weight, 
                            "NumTrials": numtrials})


if __name__ == "__main__":

    # Assumes executions is done with the cmd:
    # python randmst.py numpoints numtrials dimension
    numpoints = int(sys.argv[1])
    numtrials = int(sys.argv[2])
    dimension = int(sys.argv[3])

    with open(('./mst_dimension' + str(dimension) + '_points' +
               str(numpoints) + '.csv'), 'w') as csvfile:
        fieldnames = ["NumPoints", "Dimension", "Size", "Execution Time", 
                      "Average Weight", "NumTrials"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        run_mst(numtrials, numpoints, dimension, csvwriter)
        print("")
        print("Execution has completed and csv file made with the above\
              results in the current working directory.")

    # Loop through possible parameters for quick analysis or testing
    # with open('./mst_sizes_testing.csv', 'a') as csvfile:
    #    fieldnames = ["NumPoints", "Dimension", "Size", "Execution Time", 
    #                   "Average Weight"]
    #    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # set dimensions for 0, 2, 3, and 4
        # for dimension in [0, 2, 3, 4]:

        # set number of points in graph to the 2^n
        #    for point_power in range(8, 12):
        #        run_mst(5, int(math.pow(2, point_power)), dimension, csvwriter)
