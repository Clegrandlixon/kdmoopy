from logging import disable
import math
import random
import re
from time import time

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import RoutingSolution

"""
.. module:: MOVRPTW
    :objectives: minimization of the total cost and minimization of the total waiting time
    :platform: Unix, Windows
    :synopsis: Multi-Objective Vehicle Routing Problem

.. moduleauthor:: Clément Legrand <clement.legrand4.etu@univ-lille.fr>
"""


class MOVRPTW(PermutationProblem):
    """ Class representing a VRPTW Problem. """

    def __init__(self, granularity: int, instance: str = None , referenceFront = None, idealPoint = None, nadirPoint = None):
        """
        :param granularity: number of neighbors considered for the granularity in the local search
        :param instance: the path to the file which describes the instance
        :param referenceFront: a list of solutions to the problem. If provided, ideal and nadir are automatically computed
        :param nadirPoint: worst point, required to compute the hypervolume
        :param idealPoint: best point, required to compute the hypervolume
        """
        super(MOVRPTW, self).__init__()

        positions, distance_matrix, intervalDistance, time_matrix, waiting_matrix, intervalWaiting, number_of_customers, list_of_demands, capacity, list_ready_time, list_due_date, list_service_duration = self.__read_from_file(instance)
        
        self.positions = positions
        self.distance_matrix = distance_matrix 

        self.time_matrix = time_matrix
        self.waiting_matrix = waiting_matrix
        
        self.list_of_demands = list_of_demands
        self.capacity = capacity
        
        self.list_ready_time = list_ready_time
        self.list_due_date = list_due_date
        self.list_service_duration = list_service_duration
        
        self.granularity = granularity

        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = number_of_customers
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.metricsMatrices = [distance_matrix, waiting_matrix] # 1st matrix for the 1st objective and so on
        self.extremaMetrics = [intervalDistance, intervalWaiting]

        self.reference_front = referenceFront
        if not referenceFront is None:
            self.ideal = [min([i[0] for i in referenceFront]), min([i[1] for i in referenceFront])]
            self.nadir = [max([i[0] for i in referenceFront]), max([i[1] for i in referenceFront])]
        elif not (idealPoint is None) and not (nadirPoint is None):
            self.ideal = idealPoint
            self.nadir = nadirPoint
        else:
            self.ideal = None
            self.nadir = None

    def __read_from_file(self, filename: str):
        """
        This function reads a VRPTW Problem instance from a file.

        :param filename: File which describes the instance: either a Solomon or a Generated instance.
        """

        benchmark = filename.split('/')[2]
        if benchmark == 'Solomon':
            specific_numbers = (10, 6)
        elif benchmark == 'generated_instances':
            specific_numbers = (15, 11)
        else:
            raise ValueError('Unknown benchmark')

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]
            nbLines = len(data)
            dimension = nbLines - specific_numbers[0]

            for i in range(nbLines):
                item = data[i]
                correct_item = []
                if item.startswith('NUMBER'):
                    item = data[i+1]
                    split_item = item.split(' ')
                    for k in range(len(split_item)):
                        if split_item[k] != '' and split_item[k] != '\t' and k < len(split_item)-1:
                            correct_item.append(split_item[k])
                        elif k == len(split_item)-1 and split_item[k] != '\n':
                            correct_item.append(split_item[k][:-1])
                    nbVehicles, capacity = [int(x) for x in correct_item]
                    break

            if dimension == 50:
                dimension = 51
            elif dimension == 100:
                dimension = 101

            c = [-1.0] * (2 * dimension)
            list_of_demands = [-1] * dimension
            list_ready_time = [-1] * dimension
            list_due_date = [-1] * dimension
            list_service_duration = [-1] * dimension
            
            for i in range(specific_numbers[1], nbLines): 
                item = data[i]
                if item != "" and item[0].isdigit():
                    # customer 0 is the depot
                    split_item = item.split(' ')
                    correct_item = []

                    for k in range(len(split_item)):
                        if split_item[k] != '' and split_item[k] != '\t' and k < len(split_item)-1:
                            correct_item.append(split_item[k])
                        if k == len(split_item)-1 and split_item[k] != '\n':
                            if not split_item[k][:-1] == "":
                                correct_item.append(split_item[k][:-1])

                    j, customer_x, customer_y, demand, ready_time, due_date, service_duration = [int(x) for x in correct_item]
                    c[2 * j] = customer_x 
                    c[2 * j + 1] = customer_y
                    list_of_demands[j] = demand
                    list_ready_time[j] = ready_time
                    list_due_date[j] = due_date
                    list_service_duration[j] = service_duration

            distance_matrix = [[-1] * dimension for _ in range(dimension)]
            intervalDistances = [10**6, 0]
            time_matrix = [[-1] * dimension for _ in range(dimension)]
            waiting_matrix = [[-1] * dimension for _ in range(dimension)]  
            intervalWaiting = [10**6, 0]

            for i in range(dimension):
                distance_matrix[i][i] = 0
                time_matrix[i][i] = 0
                waiting_matrix[i][i] = 0
                for j in range(dimension):
                    dist = math.sqrt((c[i * 2] - c[j * 2]) ** 2 + (c[i * 2 + 1] - c[j * 2 + 1]) ** 2)
                    dist = int(10 * dist)/10 # distance computed as usual in the literature
                    distance_matrix[i][j] = dist
                    if dist > intervalDistances[1]:
                        intervalDistances[1] = dist
                    if dist < intervalDistances[0]:
                        intervalDistances[0] = dist

                    # here time = distance
                    time_matrix[i][j] = dist
                    
                    # waiting time
                    ready_time_i, ready_time_j = list_ready_time[i], list_ready_time[j]
                    service_duration_i = list_service_duration[i]
                    waitingTime_ij = max(0, ready_time_j - (ready_time_i + service_duration_i + dist))
                    waiting_matrix[i][j] = waitingTime_ij

                    if waitingTime_ij > intervalWaiting[1]:
                        intervalWaiting[1] = waitingTime_ij
                    if waitingTime_ij < intervalWaiting[0]:
                        intervalWaiting[0] = waitingTime_ij


            return c, distance_matrix, intervalDistances, time_matrix, waiting_matrix, intervalWaiting, dimension-1, list_of_demands, capacity, list_ready_time, list_due_date, list_service_duration


    def evaluate(self, solution: RoutingSolution) -> RoutingSolution:
        """
        Apply the split algorithm from T. Vidal
        """
         
        s = solution.variables
        s = [0] + [i+1 for i in s] # translate the variables to add the depot
        
        nbCustomers = len(s)-1
        potential = [[1.e30, (None, None)]] * (nbCustomers+1)
        pred = [-1] * (nbCustomers+1)

        potential[0] = [0, (0,0)]

        # Split algorithm
        for i in range(0, nbCustomers):
            load = 0
            distance = 0
            duration = 0
            totalTime = 0
            waitingTime = 0
            timeWarp = 0
            earliestDeparture = 0
            latestDeparture = self.list_due_date[0]
            for j in range(i+1, nbCustomers+1):
                if load <= self.capacity and timeWarp == 0:
                    
                    load += self.list_of_demands[s[j]] 
                    
                    if j == i+1:
                        # start a new route [0, s[j]]
                        delta = self.distance_matrix[s[j]][0]
                        distance += self.distance_matrix[s[j]][0]
                        duration += self.list_service_duration[s[j]] + self.time_matrix[0][s[j]]
                        totalTime += self.list_service_duration[s[j]] + self.time_matrix[0][s[j]]
                        earliestDeparture = max(earliestDeparture, self.list_ready_time[s[j]] - delta)
                        latestDeparture = min(latestDeparture, self.list_due_date[s[j]] - delta)
                        waitingTime = duration - totalTime - timeWarp
                        
                    else:
                        # add the customer s[j] to the current route
                        delta = duration + timeWarp + self.time_matrix[s[j-1]][s[j]]
                        delta_WT = max(self.list_ready_time[s[j]] - delta - latestDeparture, 0)
                        delta_TW = max(earliestDeparture + delta - self.list_due_date[s[j]], 0)
                        
                        duration += self.list_service_duration[s[j]] + self.time_matrix[s[j-1]][s[j]] + delta_WT
                        timeWarp += delta_TW
                        earliestDeparture = max(self.list_ready_time[s[j]] - delta, earliestDeparture) - delta_WT
                        latestDeparture = min(self.list_due_date[s[j]] - delta, latestDeparture) + delta_TW
                        distance += self.distance_matrix[s[j-1]][s[j]]
                        totalTime += self.list_service_duration[s[j]] + self.time_matrix[s[j-1]][s[j]]
                        waitingTime = duration - totalTime - timeWarp

                    # the cost depends on the aggregations used for the solution
                    cost = solution.attributes["weights"][0]*(distance + self.distance_matrix[s[j]][0]) + solution.attributes["weights"][1]*(waitingTime)
                    
                    if (potential[i][0] + cost < potential[j][0] and load <= self.capacity and timeWarp == 0):  # verify that the route still satisfies the constraints
                        # timewarp == 0 prevents any time window violation 
                        potential[j] = [potential[i][0] + cost, (potential[i][1][0] + distance + self.distance_matrix[s[j]][0], potential[i][1][1] + waitingTime)]
                        pred[j] = i
                    if timeWarp > 0:
                        break
                
                else:
                    # no possibility to take the next customer so we break the route
                    break
            
        # Core of split finished
        # Now sweeping the route in O(n) to report the solution
        if potential[nbCustomers][0] > 1.e29:
            print("ERROR: no Split solution has been propagated until the last node")
            raise ArithmeticError("ERROR : no Split solution has been propagated until the last node")

        # Counting the number of routes using pred structure (linear)
        solutionNbRoutes = 0
        cour = nbCustomers
        while cour != 0:
            cour = pred[cour]
            solutionNbRoutes += 1

        # filling listRoutes in the good order (linear)
        cour = nbCustomers
        listRoutes  = [-1] * solutionNbRoutes
        for i in range(solutionNbRoutes-1, -1, -1):
            cour = pred[cour]
            listRoutes[i] = cour+1
          
        completeListRoutes = []
        
        for i in range(solutionNbRoutes):
            if i < solutionNbRoutes-1:
                start = listRoutes[i]
                end = listRoutes[i+1]
                currentRoute = [0] + s[start:end] + [0]
                completeListRoutes.append(currentRoute)
            else:
                start = listRoutes[i]
                currentRoute = [0] + s[start:] + [0]
                completeListRoutes.append(currentRoute)
  
        # By construction the split function is always feasible
        solution.objectives[0] = round(potential[nbCustomers][1][0], 2)
        solution.objectives[1] = round(potential[nbCustomers][1][1], 2)
        solution.structure = completeListRoutes

        return solution

    def is_arc_infeasible(self, arc):
        """
        Either incompatible TW or too high demand 
        """
        i,j = arc
        time_i = self.list_ready_time[i] + self.list_service_duration[i] + self.time_matrix[i][j]
        time_j = self.list_due_date[j]
        return time_i > time_j or self.list_of_demands[i] + self.list_of_demands[j] > self.capacity

    def init_sequence(self, customer):
        sequence_attributes = {}
        sequence_attributes['D'] = self.list_service_duration[customer]
        sequence_attributes['TW'] = 0
        sequence_attributes['E'] = self.list_ready_time[customer]
        sequence_attributes['L'] = self.list_due_date[customer]
        sequence_attributes['C'] = 0
        sequence_attributes['Q'] = self.list_of_demands[customer] 
        sequence_attributes['nQ'] = max(sequence_attributes['Q'] - self.capacity, 0)
        sequence_attributes['WT'] = 0
        sequence_attributes['T'] = self.list_service_duration[customer]
        return sequence_attributes

    def compute_attributes_pattern(self, pattern):
        attributes_pattern = self.init_sequence(pattern[0])
        current_pattern = [pattern[0]]
        n = len(pattern)
        for i in range(1, n):
            next_customer = pattern[i]
            sequence_customer = self.init_sequence(next_customer)
            attributes_pattern = self.concatenate_subsequences((current_pattern[0], current_pattern[-1]), attributes_pattern, (next_customer, next_customer), sequence_customer)
            current_pattern.append(next_customer)
        return attributes_pattern

    def is_route_feasible(self, route):
        sequenceRoute = self.compute_attributes_pattern(route)
        return sequenceRoute['TW'] == 0 or sequenceRoute['Q'] <= self.capacity

    def get_names_objectives(self):
        return ['C', 'WT']

    def get_tour(self, solution):
        return [0] + [i+1 for i in solution.variables]
 
    def add_depot_to_sequence(self, subsequence, attributes_subsequence, solution):
        """
        Update the subsequence given in parameter with the depot (left and right).
        :param subsequence: a tuple that contains the starting and ending point of the sequence
        :param attributes_subsequence: The attributes of the subsequence
        :param solution: A solution (required to obtain the attributes of the depot)
        """
        attributes_depot = solution.sequences[0][0]
        attributes = self.concatenate_subsequences((0,0), attributes_depot, subsequence, attributes_subsequence)
        attributes = self.concatenate_subsequences((0,subsequence[-1]), attributes, (0,0), attributes_depot)
        return attributes

    def add_depot_to_left(self, subsequence, attributes_subsequence, solution):
        """
        Update the subsequence given in parameter with the depot (left only).
        :param subsequence: The subsequence to update
        :param attributes_subsequence: The attributes of the subsequence
        :param solution: A solution (required to obtain the attributes of the depot)
        """
        attributes_depot = solution.sequences[0][0]
        attributes = self.concatenate_subsequences((0,0), attributes_depot, subsequence, attributes_subsequence)
        return attributes
    
    def add_depot_to_right(self, subsequence, attributes_subsequence, solution):
        """
        Update the subsequence given in parameter with the depot (right only).
        :param subsequence: The subsequence to update
        :param attributes_subsequence: The attributes of the subsequence
        :param solution: A solution (required to obtain the attributes of the depot)
        """
        attributes_depot = solution.sequences[0][0]
        attributes = self.concatenate_subsequences(subsequence, attributes_subsequence, (0,0), attributes_depot)
        return attributes
        
    def concatenate_subsequences(self, subsequence1: tuple, attributes_subsequence1, subsequence2: tuple, attributes_subsequence2):
        """
        Compute the attributes of sequence: Sequence1 + Sequence2, knowing the attributes of the two sequences.
        :param subsequence1: A tuple which contains the starting and ending customer of the first sequence (no depot). 
        :param attributes_subsequence1: Dictionary that contains the attributes of the first sequence.
        :param subsequence2: A tuple which contains the starting and ending customer of the second sequence (no depot).
        :param attributes_subsequence1: Dictionary that contains the attributes of the second sequence. 
        """
        time = self.time_matrix[subsequence1[1]][subsequence2[0]]
        cost = self.distance_matrix[subsequence1[1]][subsequence2[0]] 

        delta = attributes_subsequence1['D'] + attributes_subsequence1['TW'] + time
        delta_WT = max(attributes_subsequence2['E'] - delta - attributes_subsequence1['L'], 0)
        delta_TW = max(attributes_subsequence1['E'] + delta - attributes_subsequence2['L'], 0)
        
        sequence_attributes = {}

        sequence_attributes['D'] = attributes_subsequence1['D'] + attributes_subsequence2['D'] + time + delta_WT
        sequence_attributes['TW'] = attributes_subsequence1['TW'] + attributes_subsequence2['TW'] + delta_TW
        sequence_attributes['E'] = max(attributes_subsequence2['E'] - delta, attributes_subsequence1['E']) - delta_WT
        sequence_attributes['L'] = min(attributes_subsequence2['L'] - delta, attributes_subsequence1['L']) + delta_TW
        sequence_attributes['C'] = attributes_subsequence1['C'] + attributes_subsequence2['C'] + cost
        sequence_attributes['Q'] = attributes_subsequence1['Q'] + attributes_subsequence2['Q']
        sequence_attributes['nQ'] = max(sequence_attributes['Q'] - self.capacity, 0)
        sequence_attributes['T'] = attributes_subsequence1['T'] + attributes_subsequence2['T'] + time

        # D contains the total travel time, the time warp and the waiting time
        sequence_attributes['WT'] = sequence_attributes['D'] - sequence_attributes['T'] - sequence_attributes['TW']
        return sequence_attributes
        
    def formating_tour(self, tour):
        """ Deletes the depot from tour """
        if tour[0] == 0 and tour[-1] == 0:
            return tour[1:-1]
        elif tour[0] == 0:
            return tour[1:]
        elif tour[-1] == 0:
            return tour[:-1]
        else:
            return tour

    def compute_subsequences(self, tour, solution: RoutingSolution, reverse: bool):
        """
        Compute the subsequences associated to the (partial) tour given.
        :param tour: A subset of the permutation.
        :param solution: The solution from which comes the tour.
        :param reverse: If True, then it also computes the reversed sequences. 
        """
        length = len(tour)
        for size_sequence in range(1, length+1):
            for i in range(length-size_sequence+1): 
                detailed_sequence = tour[i:i+size_sequence]
                sequence = (detailed_sequence[0], detailed_sequence[-1])
                if size_sequence == 1: # initialize the sequences
                    customer = sequence[0] 
                    sequence_attributes = {}
                    sequence_attributes['D'] = self.list_service_duration[customer]
                    sequence_attributes['TW'] = 0
                    sequence_attributes['E'] = self.list_ready_time[customer]
                    sequence_attributes['L'] = self.list_due_date[customer]
                    sequence_attributes['C'] = 0
                    sequence_attributes['Q'] = self.list_of_demands[customer] 
                    sequence_attributes['nQ'] = max(sequence_attributes['Q'] - self.capacity, 0)
                    sequence_attributes['WT'] = 0
                    sequence_attributes['T'] = self.list_service_duration[customer]

                else:
                    subsequence1 = (detailed_sequence[0], detailed_sequence[-2])
                    subsequence2 = (detailed_sequence[-1], detailed_sequence[-1])
                    if reverse:
                        attributes_subsequence1 = solution.reversed_sequences[subsequence1[0]][subsequence1[-1]]
                        attributes_subsequence2 = solution.reversed_sequences[subsequence2[0]][subsequence2[-1]]
                    else:
                        attributes_subsequence1 = solution.sequences[subsequence1[0]][subsequence1[-1]]
                        attributes_subsequence2 = solution.sequences[subsequence2[0]][subsequence2[-1]]
                    sequence_attributes = self.concatenate_subsequences(subsequence1, attributes_subsequence1, subsequence2, attributes_subsequence2)
                    
                if reverse:
                    solution.reversed_sequences[sequence[0]][sequence[1]] = sequence_attributes
                else:
                    solution.sequences[sequence[0]][sequence[1]] = sequence_attributes

    def create_solution(self) -> RoutingSolution:
        new_solution = RoutingSolution(number_of_variables=self.number_of_variables, number_of_objectives=self.number_of_objectives)
        new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)

        return new_solution

    @property
    def number_of_cities(self):
        return self.number_of_variables

    def get_name(self):
        return "Multi-Objective VRPTW"
        