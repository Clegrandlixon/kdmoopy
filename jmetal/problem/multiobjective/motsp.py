import math
import random
import re

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution

"""
.. module:: TSP
   :platform: Unix, Windows
   :synopsis: Single Objective Traveling Salesman problem

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class BOTSP(PermutationProblem):
    """ Class representing TSP Problem. """

    def __init__(self, granularity: int, instance1: str = None, instance2: str = None, reference_front = None, initializationFile = None):
        super(BOTSP, self).__init__()

        positions1, distance_matrix1, interval1, number_of_cities1 = self.__read_from_file(instance1)
        positions2, distance_matrix2, interval2, number_of_cities2 = self.__read_from_file(instance2)

        if number_of_cities1 != number_of_cities2:
            raise ValueError("Instance1 and Instance2 do not have the same number of cities")
        
        self.positions = positions1

        self.distance_matrix1 = distance_matrix1
        self.distance_matrix2 = distance_matrix2

        self.metricsMatrices = [distance_matrix1, distance_matrix2]
        self.extremaMetrics = [interval1, interval2]

        self.granularity = granularity

        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = number_of_cities1-1
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.reference_front = reference_front
        self.initialSolutions = self.__read_file_PPLS(initializationFile)

        if not reference_front is None:
            self.ideal = [min([i[0] for i in reference_front]), min([i[1] for i in reference_front])]
            self.nadir = [max([i[0] for i in reference_front]), max([i[1] for i in reference_front])]

    def __read_from_file(self, filename: str):
        """
        This function reads a TSP Problem instance from a file.

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]

            dimension = re.compile(r'[^\d]+')

            for item in data:
                if item.startswith('DIMENSION'):
                    dimension = int(dimension.sub('', item))
                    break

            c = [-1.0] * (2 * dimension)

            for item in data:
                if item[0].isdigit():
                    j, city_a, city_b = [int(x.strip()) for x in item.split(' ')]
                    c[2 * (j - 1)] = city_a
                    c[2 * (j - 1) + 1] = city_b

            matrix = [[-1] * dimension for _ in range(dimension)]
            interval = [10**6, 0]

            for k in range(dimension):
                matrix[k][k] = 0

                for j in range(k + 1, dimension):
                    dist = math.sqrt((c[k * 2] - c[j * 2]) ** 2 + (c[k * 2 + 1] - c[j * 2 + 1]) ** 2)
                    dist = round(dist)
                    matrix[k][j] = dist
                    matrix[j][k] = dist
                    if dist > interval[1]:
                        interval[1] = dist
                    if dist < interval[0]:
                        interval[0] = dist


            return c,matrix, interval, dimension

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness1 = 0
        fitness2 = 0
        complete_tour = self.get_tour(solution) + [0]
        for i in range(self.number_of_variables+1):
            x = complete_tour[i]
            y = complete_tour[i + 1]

            fitness1 += self.distance_matrix1[x][y]
            fitness2 += self.distance_matrix2[x][y]
        
        solution.objectives[0] = fitness1
        solution.objectives[1] = fitness2

        
        solution.structure = [complete_tour.copy()]

        return solution

    def get_names_objectives(self):
        return ['cost1', 'cost2']

    def get_tour(self, solution):
        return [0] + [i+1 for i in solution.variables]

    def add_depot_to_sequence(self, subsequence, attributes_subsequence, solution):
        """
        Update the subsequence given in parameter with the depot (left and right).
        :param subsequence: The subsequence to update
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

        cost1 = self.distance_matrix1[subsequence1[1]][subsequence2[0]]
        cost2 = self.distance_matrix2[subsequence1[1]][subsequence2[0]] 

        sequence_attributes = {}

        sequence_attributes['cost1'] = attributes_subsequence1['cost1'] + attributes_subsequence2['cost1'] + cost1
        sequence_attributes['cost2'] = attributes_subsequence1['cost2'] + attributes_subsequence2['cost2'] + cost2

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

    def compute_subsequences(self, tour, solution: PermutationSolution, reverse: bool):
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
                    sequence_attributes = {}
                    sequence_attributes['cost1'] = 0
                    sequence_attributes['cost2'] = 0

                else:
                    subsequence1 = (detailed_sequence[0], detailed_sequence[-2])
                    subsequence2 = (detailed_sequence[-1], detailed_sequence[-1])
                    if reverse:
                        attributes_subsequence1 = solution.reverted_sequences[subsequence1[0]][subsequence1[-1]]
                        attributes_subsequence2 = solution.reverted_sequences[subsequence2[0]][subsequence2[-1]]
                    else:
                        attributes_subsequence1 = solution.sequences[subsequence1[0]][subsequence1[-1]]
                        attributes_subsequence2 = solution.sequences[subsequence2[0]][subsequence2[-1]]
                    sequence_attributes = self.concatenate_subsequences(subsequence1, attributes_subsequence1, subsequence2, attributes_subsequence2)
                    
                if reverse:
                    solution.reverted_sequences[sequence[0]][sequence[1]] = sequence_attributes
                else:
                    solution.sequences[sequence[0]][sequence[1]] = sequence_attributes

    def create_solution(self, weights = None) -> PermutationSolution:
        "If a file is provided, initial solutions are read from it"
        "(MOEAD only: use the weights to find the best solution associated in the file)"
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                               number_of_objectives=self.number_of_objectives)
        new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)
        self.evaluate(new_solution)
        return new_solution

    def __read_file_PPLS(self, initializationFile):
        listOfSolutions = []
        if initializationFile is None:
            return listOfSolutions
        with open(initializationFile) as file:
            print("Reading the initialization file...")
            lines = file.readlines()
            data = [line.lstrip() for line in lines if line != ""]
            cpt = 0
            values = {}
            for item in data[1:]:
                if cpt%2 == 0:
                    objectives = [int(i) for i in item.split(' ')[:-1]]
                    values['objectives'] = objectives
                else:
                    variables = [int(i)-1 for i in item.split(' ')[:-1]]
                    values['variables'] = variables[1:]
                    listOfSolutions.append(values)
                    values = {}
                cpt = (cpt+1)%2
        return listOfSolutions            
            


        

    @property
    def number_of_cities(self):
        return self.number_of_variables

    def get_name(self):
        return 'Symmetric bi-objective TSP'
