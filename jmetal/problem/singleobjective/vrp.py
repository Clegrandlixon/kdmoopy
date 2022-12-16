import math
import random
import re

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution

"""
.. module:: VRP
    :platform: Unix, Windows
    :synopsis: Single Objective Vehicle Routing Problem

.. moduleauthor:: Clément Legrand <clement.legrand-lixon@ens-rennes.fr>
"""


class VRP(PermutationProblem):
    """ Class representing VRP Problem. """

    def __init__(self, instance: str = None):
        super(VRP, self).__init__()

        distance_matrix, number_of_customers, list_of_demands, capacity = self.__read_from_file(instance)
        
        self.distance_matrix = distance_matrix
        self.list_of_demands = list_of_demands
        self.capacity = capacity

        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = number_of_customers
        self.number_of_objectives = 1
        self.number_of_constraints = 0 #1 # we only consider the capacity as a constraint

    def __read_from_file(self, filename: str):
        """
        This function reads a VRP Problem instance from a file.

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]

            dimension = re.compile(r'[^\d]+')
            capacity = re.compile(r'[^\d]+')

            for item in data:
                if item.startswith('DIMENSION'):
                    dimension = int(dimension.sub('', item))
                elif item.startswith('CAPACITY'):
                    capacity = int(capacity.sub('', item))
                    break

            c = [-1.0] * (2 * dimension)
            list_of_demands = [-1] * dimension

            for item in data:
                if item.startswith('NODE_COORD_SECTION'):
                    read_coord = True
                    read_demand = False
                if item.startswith('DEMAND_SECTION'):
                    read_coord = False
                    read_demand = True
                if item.startswith('DEPOT_SECTION'):
                    break

                if item[0].isdigit():
                    if read_coord:
                        # customer 0 is the depot
                        j, customer_x, customer_y = [int(x.strip()) for x in item.split(' ')]
                        c[2 * (j - 1)] = customer_x
                        c[2 * (j - 1) + 1] = customer_y

                    elif read_demand:
                        # demand 0 is 0 (depot)
                        j, demand_j = [int(x.strip()) for x in item.split(' ')[:2]]
                        list_of_demands[j-1] = demand_j

            matrix = [[-1] * dimension for _ in range(dimension)]

            for k in range(dimension):
                matrix[k][k] = 0

                for j in range(k + 1, dimension):
                    dist = math.sqrt((c[k * 2] - c[j * 2]) ** 2 + (c[k * 2 + 1] - c[j * 2 + 1]) ** 2)
                    dist = round(dist)
                    matrix[k][j] = dist
                    matrix[j][k] = dist

            # remove the depot from the variables
            #print("Dimension: ", dimension-1)
            #print("Capacity: ", capacity)
            return matrix, dimension-1, list_of_demands, capacity

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        """
        Apply the split algorithm of Vidal
        to evaluate a solution
        """
        # TEST with: s = [21, 31, 19, 17, 13, 7, 26, 12, 1, 16, 30, 27, 24, 29, 18, 8, 9, 22, 15, 10, 25, 5, 20, 14, 28, 11, 4, 23, 3, 2, 6] : solution of A-n32-k5
        # We should find: fitness = 784 and SolutionNbRoutes = 5
        # Test passed
         
        s = solution.variables
        print("Solution: ", s)
        s = [0] + [i+1 for i in s] # s with right indexes
        # Initialisation
        nbCustomers = self.number_of_variables
        potential = [1.e30] * (nbCustomers+1)
        pred = [-1] * (nbCustomers+1)

        potential[0] = 0

        # Split algorithm
        for i in range(0, nbCustomers):
            load = 0
            distance = 0
            for j in range(i+1, nbCustomers+1):
                if load <= self.capacity:
                    load += self.list_of_demands[s[j]]
                    if j == i+1:
                        distance += self.distance_matrix[s[j]][0]
                    else:
                        distance += self.distance_matrix[s[j-1]][s[j]]
                    cost = distance + self.distance_matrix[s[j]][0]
                    if (potential[i] + cost < potential[j] and load <= self.capacity):
                        potential[j] = potential[i] + cost
                        pred[j] = i
            
        # Core of split finished
        # Now sweeping the route in O(n) to report the solution

        if potential[nbCustomers] > 1.e29:
            print("ERROR: no Split solution has been propagated until the last node")
            raise ArithmeticError("ERROR : no Split solution has been propagated until the last node")

        # Counting the number of routes using pred structure (linear)
        solutionNbRoutes = 0
        cour = nbCustomers
        while cour != 0:
            cour = pred[cour]
            solutionNbRoutes += 1

        print("Nb routes: ", solutionNbRoutes)
        # filling listRoutes in the good order (linear)
        cour = nbCustomers
        listRoutes  = [-1] * solutionNbRoutes
        for i in range(solutionNbRoutes-1, -1, -1):
            cour = pred[cour]
            listRoutes[i] = cour+1 # indice où début la route i dans la solution
                
            
        # Remark: the split solution is always feasible ? -> no constraints are needed ?
        print("Evaluate cost: ", potential[nbCustomers])
        print()
        solution.objectives[0] = potential[nbCustomers]
        return solution


    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables, number_of_objectives=self.number_of_objectives)
        new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)

        return new_solution


    @property
    def number_of_cities(self):
        return self.number_of_variables

    def get_name(self):
        return "Symmetric VRP"
        