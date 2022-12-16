import math
import random
import re

from jmetal.core.problem import PermutationProblem, RoutingProblem
from jmetal.core.solution import PermutationSolution, RoutingSolution

"""
.. module:: VRPTW
    :platform: Unix, Windows
    :synopsis: Single Objective Vehicle Routing Problem

.. moduleauthor:: Clément Legrand <clement.legrand-lixon@ens-rennes.fr>
"""


class VRPTW(PermutationProblem):
    """ Class representing VRP Problem. """

    def __init__(self, instance: str = None):
        super(VRPTW, self).__init__()

        print("Lecture du fichier en cours...")
        distance_matrix, number_of_customers, list_of_demands, capacity, list_ready_time, list_due_date, list_service_duration = self.__read_from_file(instance)
        print("Fin de la lecture du fichier!")

        self.distance_matrix = distance_matrix
        self.list_of_demands = list_of_demands
        self.capacity = capacity
        self.list_ready_time = list_ready_time
        self.list_due_date = list_due_date
        self.list_service_duration = list_service_duration

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
            nbLines = len(data)
            dimension = nbLines - 9

            for i in range(nbLines):
                item = data[i]
                correct_item = []
                if item.startswith('NUMBER'):
                    item = data[i+1]
                    split_item = item.split(' ')
                    for k in range(len(split_item)):
                        if split_item[k] != '' and k < len(split_item)-1:
                            correct_item.append(split_item[k])
                        if k == len(split_item)-1:
                            correct_item.append(split_item[k][:-1])
                    nbVehicles, capacity = [int(x) for x in correct_item]
                    break

            c = [-1.0] * (2 * dimension)
            list_of_demands = [-1] * dimension
            list_ready_time = [-1] * dimension
            list_due_date = [-1] * dimension
            list_service_duration = [-1] * dimension

            for i in range(6, nbLines):
                item = data[i]
                if item != "" and item[0].isdigit():
                    # customer 0 is the depot
                    split_item = item.split(' ')
                    correct_item = []
                    for k in range(len(split_item)):
                        if split_item[k] != '' and k < len(split_item)-1:
                            correct_item.append(split_item[k])
                        if k == len(split_item)-1:
                            if not split_item[k][:-1] == "":
                                correct_item.append(split_item[k][:-1])
                    j, customer_x, customer_y, demand, ready_time, due_date, service_duration = [int(x) for x in correct_item]
                    c[2 * j] = customer_x
                    c[2 * j + 1] = customer_y
                    list_of_demands[j] = demand
                    list_ready_time[j] = ready_time
                    list_due_date[j] = due_date
                    list_service_duration[j] = service_duration

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
            return matrix, dimension-1, list_of_demands, capacity, list_ready_time, list_due_date, list_service_duration

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        """
        Apply the split algorithm of Vidal
        to evaluate a solution
        """
        # other objective: late time for time windows
         
        s = solution.variables
        #print("Solution: ", s)
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
            listRoutes[i] = cour+1 # indice o� début la route i dans la solution
                
            
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
        