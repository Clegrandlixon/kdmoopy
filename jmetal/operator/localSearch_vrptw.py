import copy
from curses import A_ALTCHARSET
from operator import mod
import random
import time
from typing import List

from jmetal.core.operator import boVRPTWLocalSearch
from jmetal.core.solution import Solution,  PermutationSolution
from jmetal.operator import crossover
from jmetal.problem.multiobjective.movrptw import MOVRPTW
from jmetal.util.ckecking import Check
from jmetal.core.problem import Problem
from jmetal.util.neighborhood import Neighborhood

"""
.. module:: local search operators for routing problems
   :platform: Unix, Windows
   :synopsis: Module implementing local search operators.

.. moduleauthor:: Clément Legrand <clement.legrand4.etu@univ-lille.fr>
"""

class TwoOptStar(boVRPTWLocalSearch):
    """ Exchange the tails of two distinct routes """

    def generate_candidates(self, solution: PermutationSolution):
        """ Generate the list of possible candidates to start a move 
        
        :param solution: The solution that undergoes the local search
        :return: A list containing the candidates that can move
        """
        routes = solution.structure
        candidates = []
        for i in range(len(routes)):
            current_route = routes[i]
            candidates += [(i,j) for j in range(len(current_route)-1) if not current_route[j] in solution.forbidden_elements] # possible to remove the depot
        random.shuffle(candidates)
        return candidates

    def generate_neighborhood_candidate(self, candidate: tuple, solution: PermutationSolution):
        """ Generate the neighborhood of a candidate

        :param candidate: A tuple (index of route, index of customer)
        :param solution: The solution that undergoes the local search
        :return: A list containing the neghbours of the candidate
        """
        routes = solution.structure
        neighborhood = []
        for i in range(len(routes)):
            if i != candidate[0]:
                current_route = routes[i]
                neighborhood += [(i,j) for j in range(len(current_route)-1) if self.neighbours[solution.associatedProblem][candidate[1]][current_route[j]] and not current_route[j] in solution.forbidden_elements]
        random.shuffle(neighborhood)
        return neighborhood

    def compute_information_move(self, element1: tuple, element2: tuple, solution: PermutationSolution):
        (indexR1, arc1) = element1
        (indexR2, arc2) = element2
        routes = solution.structure
        route1, route2 = routes[indexR1], routes[indexR2]
        
        list_sequences1 = [(route1[1:arc1+1], False), (route2[arc2+1:-1], False)]
        list_sequences2 = [(route2[1:arc2+1], False), (route1[arc1+1:-1], False)]
        
        new_route1, attributes_newR1 = self.merge_Nsequences(list_sequences1, solution)
        new_route2, attributes_newR2 = self.merge_Nsequences(list_sequences2, solution)
        return new_route1, attributes_newR1, new_route2, attributes_newR2
    
    def get_name(self) -> str:
        return "2-opt*"

class Swap(boVRPTWLocalSearch):
    def generate_candidates(self, solution: PermutationSolution):
        """ Generate the list of possible candidates to start a move 
        
        :param solution: The solution that undergoes the local search
        :return: A list containing the candidates that can move
        """
        routes = solution.structure
        elements = []
        for r in range(len(routes)):
            current_route = routes[r]
            elements += [(r, i) for i in range(1, len(current_route)-1) if not current_route[i] in solution.forbidden_elements]
        random.shuffle(elements)
        return elements

    def generate_neighborhood_candidate(self, candidate: tuple, solution: PermutationSolution):
        """ Generate the neighborhood of a candidate

        :param candidate: A tuple (index of route, index of customer)
        :param solution: The solution that undergoes the local search
        :return: A list containing the neghbours of the candidate
        """
        routes = solution.structure
        neighbours = []
        for r in range(len(routes)):
            current_route = routes[r]
            neighbours += [(r, i) for i in range(1, len(current_route)-1) if self.neighbours[solution.associatedProblem][candidate[1]][current_route[i]] and (r,i) != candidate and not current_route[i] in solution.forbidden_elements]
        random.shuffle(neighbours)
        return neighbours

    def compute_information_move(self, element1: tuple, element2: tuple, solution: PermutationSolution):
        routes = solution.structure
        (index_r1, index_u) = element1
        (index_r2, index_v) = element2

        route1 = routes[index_r1]
        route2 = routes[index_r2]

        if index_r1 == index_r2:
            min_index = min(index_v, index_u)
            max_index = max(index_v, index_u)
            if route1[min_index+1:max_index] == []:
                list_sequences1 = [(route1[1:min_index], False), ([route1[max_index]], False), ([route1[min_index]], False), (route1[max_index+1:-1], False)]
            else:
                list_sequences1 = [(route1[1:min_index], False), ([route1[max_index]], False), (route1[min_index+1:max_index], False), ([route1[min_index]], False), (route1[max_index+1:-1], False)]

            new_route1, attributes_newR1 = self.merge_Nsequences(list_sequences1, solution)
            new_route2 = None
            attributes_newR2 = None
        else:
            list_sequences1 = [(route1[1:index_u], False), ([route2[index_v]], False), (route1[index_u+1:-1], False)]
            list_sequences2 = [(route2[1:index_v], False), ([route1[index_u]], False), (route2[index_v+1:-1], False)]
            new_route1, attributes_newR1 = self.merge_Nsequences(list_sequences1, solution)
            new_route2, attributes_newR2 = self.merge_Nsequences(list_sequences2, solution)

        return new_route1, attributes_newR1, new_route2, attributes_newR2 

    def get_name(self) -> str:
        return "Swap"

class Relocate(boVRPTWLocalSearch):
    def generate_candidates(self, solution: PermutationSolution):
        """ Generate the list of possible candidates to start a move 
        
        :param solution: The solution that undergoes the local search
        :return: A list containing the candidates that can move
        """
        routes = solution.structure
        elements = []
        for r in range(len(routes)):
            current_route = routes[r]
            elements += [(r, i) for i in range(1, len(current_route)-1) if not current_route[i] in solution.forbidden_elements]

        random.shuffle(elements)
        return elements

    def generate_neighborhood_candidate(self, element: tuple, solution: PermutationSolution):
        """ Generate the neighborhood of a candidate

        :param candidate: A tuple (index of route, index of customer)
        :param solution: The solution that undergoes the local search
        :return: A list containing the neghbours of the candidate
        """
        routes = solution.structure
        neighbours = []
        for r in range(len(routes)):
            current_route = routes[r]
            if r != element[0]:
                neighbours += [(r, i) for i in range(1, len(current_route)-1) if self.neighbours[solution.associatedProblem][element[1]][current_route[i]] and not current_route[i] in solution.forbidden_elements] #or possible_neighbours[current_route[i+1]]]
            else:
                neighbours += [(r, i) for i in range(1, len(current_route)-1) if self.neighbours[solution.associatedProblem][element[1]][current_route[i]] and (r, i+1) != element and (r, i) != element and len(current_route) > 3 and not current_route[i] in solution.forbidden_elements]
        random.shuffle(neighbours)
        return neighbours

    def compute_information_move(self, element1: tuple, element2: tuple, solution: PermutationSolution):
        routes = solution.structure
        (index_r1, index_u) = element1
        (index_r2, index_v) = element2
        
        route1 = routes[index_r1]
        route2 = routes[index_r2]

        if index_r1 == index_r2:
            if index_v < index_u:
                list_sequences1 = [(route1[1:index_v+1], False), ([route1[index_u]], False), (route1[index_v+1:index_u], False), (route1[index_u+1:-1], False)]
            else:
                list_sequences1 = [(route1[1:index_u], False), (route1[index_u+1:index_v+1], False), ([route1[index_u]], False), (route1[index_v+1:-1], False)]

            new_route1, attributes_newR1 = self.merge_Nsequences(list_sequences1, solution)
            new_route2 = None
            attributes_newR2 = None

        else:
            list_sequences1 = [(route1[1:index_u], False), (route1[index_u+1:-1], False)]
            list_sequences2 = [(route2[1:index_v+1], False), ([route1[index_u]], False), (route2[index_v+1:-1], False)]
            new_route1, attributes_newR1 = self.merge_Nsequences(list_sequences1, solution)
            new_route2, attributes_newR2 = self.merge_Nsequences(list_sequences2, solution)
        return new_route1, attributes_newR1, new_route2, attributes_newR2 

    def get_name(self) -> str:
        return "Relocate"

class ApplyManyOperators(TwoOptStar, Swap, Relocate):
    """ A class that applies the three operators: 2opt*, Swap and Relocate """

    def __init__(self, problem, probability, strategy, neighbours) -> None:
        super(ApplyManyOperators, self).__init__(problem, neighbours, probability, strategy)


    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        x = random.random()
        
        if x > self.probability:
            return solution

        twoOptStar = TwoOptStar(self.problem, self.neighbours)
        swap = Swap(self.problem, self.neighbours)
        relocate = Relocate(self.problem, self.neighbours)

        operators = [twoOptStar, swap, relocate]
        # Randomize the order of the operators
        random.shuffle(operators)

        frequency_improvement = {}
        iterations_operator = {}
        for operator in operators:
            frequency_improvement[operator.get_name()] = 0
            iterations_operator[operator.get_name()] = 0

        best_fitness = solution.objectives[0] * solution.attributes["weights"][0] + solution.objectives[1] * solution.attributes["weights"][1]

        for operator in operators:
            solution.improvedByLS = True
            nb_iteration = 0
            time_ope = time.time()
            size_neighborhood = 0
    
            while solution.improvedByLS and nb_iteration < 1000: # nb_iteration is bounded to avoid inifite loop
                nb_iteration += 1
                best_move = None
                solution.improvedByLS = False

                elements_to_consider = operator.generate_candidates(solution)
                k = len(elements_to_consider)
                cpt_out = 0

                if self.strategy == "FIRST-BEST":
                    # Move one customer to its best place 
                    while not solution.improvedByLS and cpt_out < k:
                        current_element = elements_to_consider[cpt_out] 
                        cpt_out += 1

                        neighborhood_element = operator.generate_neighborhood_candidate(current_element, solution)
                        
                        size_neighborhood += len(neighborhood_element)
                        for neighbor in random.sample(neighborhood_element, len(neighborhood_element)):
                            move = (current_element, neighbor)
                            result = operator.execute(move, solution)
                            if result != (None, None, None):
                                index_routes_modified, new_routes, objectives = result
                                new_fitness = solution.attributes["weights"][0] * objectives[0] + solution.attributes["weights"][1] * objectives[1]
                                if round(new_fitness, 2) < round(best_fitness, 2):
                                    best_fitness = new_fitness
                                    best_move = result

                        # Apply the best move found (for the considered customer)
                        if best_move != None:
                            solution = operator.apply_move(best_move[0], best_move[1], best_move[2], solution)
                            frequency_improvement[operator.get_name()] += 1
                
                elif self.strategy == "BEST-BEST":
                    # Find the best possible move
                    for current_element in elements_to_consider:
                        neighborhood_element = operator.generate_neighborhood_candidate(current_element, solution)
                        size_neighborhood += len(neighborhood_element)
                        for neighbor in neighborhood_element:
                            move = (current_element, neighbor)
                            result = operator.execute(move, solution)
                            if result != (None, None, None):
                                index_routes_modified, new_routes, objectives = result
                                new_fitness = solution.attributes["weights"][0] * objectives[0] + solution.attributes["weights"][1] * objectives[1]
                                if round(new_fitness, 2) < round(best_fitness, 2):
                                    best_fitness = new_fitness
                                    best_move = result
                    # Apply the best move found
                    if best_move != None:
                        solution = operator.apply_move(best_move[0], best_move[1], best_move[2], solution)
                
            time_ope = time.time() - time_ope
            iterations_operator[operator.get_name()] = nb_iteration

        return solution

    def get_name(self) -> str:
        return "Local Search VRPTW"