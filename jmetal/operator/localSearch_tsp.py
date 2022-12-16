import copy
from operator import index, mod
import random
import time
from typing import List
import os
import sys
import shutil

from contextlib import redirect_stdout

from numpy.random.mtrand import noncentral_chisquare, rand

from jmetal.core.operator import boTSPLocalSearch
from jmetal.core.solution import Solution,  PermutationSolution
from jmetal.operator import crossover
from jmetal.problem.multiobjective.motsp import BOTSP
from jmetal.util.ckecking import Check
from jmetal.core.problem import Problem
from jmetal.util.neighborhood import Neighborhood

"""
.. module:: local search operators for the TSP
   :platform: Unix, Windows
   :synopsis: Module implementing local search operators for TSP problems.

.. moduleauthor:: Clément Legrand <clement.legrand4.etu@univ-lille.fr>
"""

class TwoOpt(boTSPLocalSearch):
    """ Class defining the 2-opt operator. 
    Two arcs from a single route are chosen, and they are exchanged. 
    This operator will work randomly with first improvement. 
    If no improvement found a limit time should be added. """
    
    def generate_elements(self, solution: PermutationSolution):
        routes = solution.structure
        elements = []
        for i in range(len(routes)):
            current_route = routes[i]
            elements += [(i,j) for j in range(1, len(current_route)-1)]
        random.shuffle(elements)
        return elements

    def generate_neighborhood_element(self, element: tuple, possible_neighbours, solution: PermutationSolution):
        """ An arc is completely characterised by the index of the route and the position where starts the arc. 
            e.g. (2,3) refers to the arc from route 2 which starts at position 3 (in the route). 
            To generate the neighborhood of the arc we need all possible tuples. """
        routes = solution.structure
        neighborhood_arc = []
        current_route = routes[element[0]]
        neighborhood_arc += [(element[0],j) for j in range(element[1]+1, len(current_route)-1) if possible_neighbours[element[1]][current_route[j]]]
        return neighborhood_arc

    def exchange_elements(self, element1: tuple, element2: tuple, solution: PermutationSolution):
        """ Exchange two arcs in the route. Return the three parts of the route. """
        
        (indexR1, arc1) = element1
        (indexR2, arc2) = element2
        routes = solution.structure
        route = routes[indexR1]
        start = route[1 : arc1 + 1]
        reverse_mid = route[arc1+1 : arc2 + 1]
        reverse_mid.reverse()
        end = route[arc2+1: -1]
        
        list_sequences1 = [(start, False), (reverse_mid, True), (end, False)]
        
        new_route1, attributes_newR1 = self.merge_Nsequences(list_sequences1, solution)
        
        return new_route1, attributes_newR1, None, None 

    def get_name(self) -> str:
        return "Two-Opt"

class TwoOptPrime(boTSPLocalSearch):
    def generate_elements(self, solution: PermutationSolution):
        route = solution.structure[0]
        elements = []
        elements += [j for j in range(1, len(route)-1)]
        random.shuffle(elements)
        return elements

    def generate_neighborhood_element(self, element: tuple, possible_neighbours, solution: PermutationSolution):
        """ An arc is completely characterised by the index of the route and the position where starts the arc. 
            e.g. (2,3) refers to the arc from route 2 which starts at position 3 (in the route). 
            To generate the neighborhood of the arc we need all possible tuples. """
        routes = solution.structure
        current_route = routes[0]
        neighborhood_arc = [j for j in range(element+1, len(current_route)-1) if possible_neighbours[element][current_route[j]]]
        return neighborhood_arc

    def execute(self, index1, index2, solution: PermutationSolution):
        route = solution.structure[0]
        start = route[: index1 + 1]
        reverse_mid = route[index1+1 : index2 + 1]
        reverse_mid.reverse()
        end = route[index2+1:]

        new_route = [start + reverse_mid + end]
        new_objectives = solution.objectives.copy()
        new_objectives[0] = new_objectives[0] - self.problem.metricsMatrices[0][route[index1]][route[index1+1]] - self.problem.metricsMatrices[0][route[index2]][route[index2+1]] + self.problem.metricsMatrices[0][route[index1]][route[index2]] + self.problem.metricsMatrices[0][route[index1+1]][route[index2+1]]
        new_objectives[1] = new_objectives[1] - self.problem.metricsMatrices[1][route[index1]][route[index1+1]] - self.problem.metricsMatrices[1][route[index2]][route[index2+1]] + self.problem.metricsMatrices[1][route[index1]][route[index2]] + self.problem.metricsMatrices[1][route[index1+1]][route[index2+1]]
        new_objectives = [round(i, 3) for i in new_objectives]  
        return 0, new_route, new_objectives

    def apply_move(self, index_route_modified: list, new_routes: list, objectives, solution):
        solution.structure = [new_routes[index_route_modified]]
        solution.objectives = objectives
        solution.variables = [i-1 for i in solution.structure[0][1:-1]]
        solution.improvedByLS = True
        return solution

    def get_name(self) -> str:
        return "TwoOpt (without sequences)"

class Swap(boTSPLocalSearch):
    def generate_elements(self, solution: PermutationSolution):
        routes = solution.structure
        elements = []
        for r in range(len(routes)):
            current_route = routes[r]
            elements += [(r, i) for i in range(1, len(current_route)-1)]
        random.shuffle(elements)
        return elements

    def generate_neighborhood_element(self, element: tuple, possible_neighbours: list, solution: PermutationSolution):
        routes = solution.structure
        neighbours = []
        for r in range(len(routes)):
            current_route = routes[r]
            neighbours += [(r, i) for i in range(1, len(current_route)-1) if possible_neighbours[element[1]][current_route[i]] and (r,i) != element]
        random.shuffle(neighbours)

        return neighbours

    def exchange_elements(self, element1: tuple, element2: tuple, solution: PermutationSolution):
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

class Relocate(boTSPLocalSearch):
    def generate_elements(self, solution: PermutationSolution):
        routes = solution.structure
        elements = []
        for r in range(len(routes)):
            current_route = routes[r]
            elements += [(r, i) for i in range(1, len(current_route)-1)]

        random.shuffle(elements)
        return elements

    def generate_neighborhood_element(self, element: tuple, possible_neighbours: list, solution: PermutationSolution):
        routes = solution.structure
        neighbours = []
        for r in range(len(routes)):
            current_route = routes[r]
            if r != element[0]:
                neighbours += [(r, i) for i in range(1, len(current_route)-1) if possible_neighbours[element[1]][current_route[i]]] #or possible_neighbours[current_route[i+1]]]
            else:
                neighbours += [(r, i) for i in range(1, len(current_route)-1) if possible_neighbours[element[1]][current_route[i]] and (r, i+1) != element and (r, i) != element and len(current_route) > 3]

        random.shuffle(neighbours)
        return neighbours

    def exchange_elements(self, element1: tuple, element2: tuple, solution: PermutationSolution):
        """ Put u after v. """
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

class ApplyManyOperators(Swap, Relocate, TwoOpt):
    """
    Apply many LS operators to reach a local optimum
    """

    def __init__(self, problem, probability, strategy, neighbours) -> None:
        super(ApplyManyOperators, self).__init__(problem, probability, strategy)
        self.adaptedNeighbours = neighbours

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        x = random.random()
        if x > self.probability: #and not solution.attributes['New']:
            return solution #, 0, None, None, 0

        swap = Swap(self.problem)
        relocate = Relocate(self.problem)
        twoOpt = TwoOptPrime(self.problem) # Two opt without sequences...

        mode_ls = self.strategy # Choose between "FIRST" and "BEST"
        #operators = [swap, relocate, twoOpt]
        operators = [twoOpt]
        random.shuffle(operators)

        for operator in operators:
            solution.improvedByLS = True
            nb_iteration = 0
            time_ope = time.time()
            
            size_neighborhood = 0
            best_fitness = solution.weights[0] * solution.objectives[0] + solution.weights[1] * solution.objectives[1]
            
            while solution.improvedByLS: #and nb_iteration < 1000:
                nb_iteration += 1
                
                best_move = None
                solution.improvedByLS = False

                elements_to_consider = operator.generate_elements(solution)
                k = len(elements_to_consider)
                cpt_out = 0

                # find the first element for which we obtain an improvement
                if mode_ls == "FIRST-BEST":
                    while not solution.improvedByLS and cpt_out < k:
                        current_element = elements_to_consider[cpt_out] 
                        cpt_out += 1
                        neighborhood_element = operator.generate_neighborhood_element(current_element, self.adaptedNeighbours, solution)
                        
                        size_neighborhood += len(neighborhood_element)

                        for neighbor in neighborhood_element:
                            result = operator.execute(current_element, neighbor, solution)
                            if result != (None, None, None):
                                index_routes_modified, new_routes, objectives = result
                                new_fitness = solution.weights[0] * objectives[0] + solution.weights[1] * objectives[1]
                                if round(new_fitness, 2) < round(best_fitness, 2):
                                    best_fitness = new_fitness
                                    best_move = result

                        # Apply the best move found
                        if best_move != None:
                            #print("TIMETEST_0:", time.time()-time_ope)
                            solution = operator.apply_move(best_move[0], best_move[1], best_move[2], solution)
                
                # find the ultimate best move
                elif mode_ls == "BEST-BEST":
                    for current_element in elements_to_consider:
                        neighborhood_element = operator.generate_neighborhood_element(current_element, self.adaptedNeighbours, solution)
                        size_neighborhood += len(neighborhood_element)
                        for neighbor in neighborhood_element:
                            result = operator.execute(current_element, neighbor, solution)
                            if result != (None, None, None):
                                index_routes_modified, new_routes, objectives = result
                                new_fitness = solution.weights[0] * objectives[0] + solution.weights[1] * objectives[1]
                                if round(new_fitness, 2) < round(best_fitness, 2):
                                    best_fitness = new_fitness
                                    best_move = result
                    if best_move != None:
                        solution = operator.apply_move(best_move[0], best_move[1], best_move[2], solution)


            time_ope = time.time() - time_ope
            print(operator.get_name(), time_ope, size_neighborhood, nb_iteration)
        return solution #, 1, None, None, 0

    def get_name(self) -> str:
        return "Local Search TSP"


