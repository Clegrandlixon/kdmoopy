from abc import ABC, abstractmethod
from types import new_class
from typing import TypeVar, Generic, List

from jmetal.core.solution import RoutingSolution
from jmetal.problem.multiobjective.movrptw import MOVRPTW
from jmetal.problem.multiobjective.motsp import BOTSP
import time

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: Operator
   :platform: Unix, Windows
   :synopsis: Templates for operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Operator(Generic[S, R], ABC):
    """ Class representing operator """

    @abstractmethod
    def execute(self, source: S) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


def check_valid_probability_value(func):
    def func_wrapper(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        res = func(self, probability)
        return res
    return func_wrapper


class Mutation(Operator[S, S], ABC):
    """ Class representing mutation operator. """

    @check_valid_probability_value
    def __init__(self, probability: float):
        self.probability = probability


class Crossover(Operator[List[S], List[R]], ABC):
    """ Class representing crossover operator. """

    @check_valid_probability_value
    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def get_number_of_parents(self) -> int:
        pass

    @abstractmethod
    def get_number_of_children(self) -> int:
        pass


class Selection(Operator[S, R], ABC):
    """ Class representing selection operator. """

    def __init__(self):
        pass

"""
.. author:: Clément Legrand <clement.legrand4.etu@univ-lille.fr 
"""

class Extraction(Operator[S, R], ABC):
    """ Class representing extraction operator. """

    def __init__(self, diversificationFactor: str):
        self.diversificationFactor = diversificationFactor

class Injection(Operator[S, S], ABC):
    """ Class representing injection operator. """

    def __init__(self, probability: float, diversificationFactor: int):
        self.probability = probability
        self.diversificationFactor = diversificationFactor

class GenericPermutationLocalSearch(Operator[S, S], ABC):
    """ Class representing a local search operator for a permutation problem. """

    def __init__(self, problem, neighbours, probability = 0, strategy = "") -> None:
        self.probability = probability
        self.strategy = strategy
        self.problem = problem # the problem is required to merge sequences
        self.neighbours = neighbours
        self.nbMerge = 0

    @abstractmethod
    def generate_candidates(self):
        """ Generate all possible candidates that can move. """
        pass

    @abstractmethod
    def generate_neighborhood_candidate(self):
        """ Generate the neighborhood of an element. """
        pass
    
    def attributes_sequence(self, t_sequence, solution: RoutingSolution):
        """ Return the attributes of the sequence given in argument.
        It uses the sequences stored in the solution. 

        :param solution: The solution that undergoes the local search
        :param t_sequence: A sequence of customers in the solution. It must not contain the depot
        :return: The list of attributes of t_sequence
        """
        
        sequence, revert = t_sequence
        indexes = (sequence[0], sequence[-1])
        if revert:
            attributes = solution.reverted_sequences[indexes[0]][indexes[1]]
        else:
            attributes = solution.sequences[indexes[0]][indexes[1]]
        return attributes

    def merge_2sequences(self, sequence1, attributesSequence1, sequence2, attributesSequence2, solution, addDepot: bool):
        """ Compute the attributes of the sequence 'sequence1 + sequence2'
        
        :param sequence1: The first (non empty) list of indexes (must not contain the depot)
        :param attributes_sequence1: The attributes of sequence1
        :param sequence2: The second (non empty) list of indexes (must not contain the depot)
        :param attributes_sequence2: The attributes of sequence2
        :param add_depot: Specify if the depot is added to the extremities of the new sequence 'sequence1 + sequence2'
        :return: A tuple (new sequence, corresponding attributes)
        """
        # assert: sequences never contain the depot and contain at least one element
        newRoute = sequence1 + sequence2
        indexesLeft = (sequence1[0], sequence1[-1])
        indexesRight = (sequence2[0], sequence2[-1])
        attributesRoute = self.problem.concatenate_subsequences(indexesLeft, attributesSequence1, indexesRight, attributesSequence2)
    
        if addDepot:
            attributesRoute = self.problem.add_depot_to_sequence((newRoute[0], newRoute[-1]), attributesRoute, solution)
        
        return newRoute, attributesRoute

    def merge_Nsequences(self, sequences, solution):
        """ Compute the attributes of the sequence obtained by concatening the elements of 'sequences' in the same order
        
        :param sequences: A list of sequences (must not contain the depot)
        :param solution: The solution that contains the sequences of listSequence
        :return: A tuple (new sequence, corresponding attributes)
        """

        # remove empty sequences:
        sequences = [i for i in sequences if i[0] != []]
        n = len(sequences)
        if n == 0:
            return [0,0], solution.sequences[0][0]
        
        if n == 1:
            (seq, revert) = sequences[0]
            mergedSequence = seq
            attributesMerged = self.attributes_sequence(sequences[0], solution)
            attributesMerged = self.problem.add_depot_to_sequence((mergedSequence[0], mergedSequence[-1]), attributesMerged, solution)
            mergedSequence = [0] + mergedSequence + [0]
            return mergedSequence, attributesMerged

        mergedSequence = sequences[0][0]
        attributesMerged = self.attributes_sequence(sequences[0], solution)

        for i in range(1, n):
            addDepot = i == n-1 # the depot is added during the last step
            newSequence = sequences[i]
            # now merge the two sequences
            attributesNewSequence = self.attributes_sequence(newSequence, solution)
            mergedSequence, attributesMerged = self.merge_2sequences(mergedSequence, attributesMerged, newSequence[0], attributesNewSequence, solution, addDepot)
        mergedSequence = [0] + mergedSequence + [0]
        return mergedSequence, attributesMerged

    @abstractmethod
    def evaluate_move(self, modifiedRoutes, solution):
        """ Evaluate if the move performed is feasible and if it improves the solution

        :param modifiedRoutes: The list of routes obtained after applying the move
        :param solution: The solution where the move is applied
        :return: A tuple (isFeasible, isImproving, deltaObjectives) where deltaObjectives contains the variation of each objective if we apply the move
        """
        pass

    def execute(self, move, solution: S) -> R:
        """ Execute the move provided
        
        :param move: A list containing the elements involved in the move
        :param solution: The solution where the move is executed
        :return: A tuple (indexes of the modified routes, the new routes, new objectives)
        """
        element1, element2 = move
        routes = solution.structure
        (index_route1, index_u) = element1
        (index_route2, index_v) = element2
        route1 = routes[index_route1].copy()
        route2 = routes[index_route2].copy()

        indexes_R1 = (route1[1], route1[-2])
        attributes_R1 = solution.sequences[indexes_R1[0]][indexes_R1[1]]
        attributes_R1 = self.problem.add_depot_to_sequence(indexes_R1, attributes_R1, solution)

        indexes_R2 = (route2[1], route2[-2])
        attributes_R2 = solution.sequences[indexes_R2[0]][indexes_R2[1]]
        attributes_R2 = self.problem.add_depot_to_sequence(indexes_R2, attributes_R2, solution)

        new_route1, attributes_newR1, new_route2, attributes_newR2 = self.compute_information_move(element1, element2, solution)
        
        modifiedRoutes= [(new_route1, attributes_R1, attributes_newR1), (new_route2, attributes_R2, attributes_newR2)]
        feasible, improved, deltaObjectives = self.evaluate_move(modifiedRoutes, solution)

        if feasible and improved:
            if new_route2 == None:
                routes_modified = [index_route1]
                new_routes = [new_route1.copy()]
            else:
                routes_modified = [index_route1, index_route2]
                new_routes = [new_route1.copy(), new_route2.copy()]
            
            objective0 = solution.objectives[0] + deltaObjectives[0]
            objective1 = solution.objectives[1] + deltaObjectives[1] 

            objectives = [objective0, objective1]
            return routes_modified, new_routes, objectives 
        return None, None, None 

    def apply_move(self, indexesModifiedRoutes: list, newRoutes: list, objectives, solution: RoutingSolution):
        """ Modify the solution with the inofmration provided in argument
        
        :param indexesRouteModified: The indexes of the modified routes
        :param newRoutes: The routes obtained after applying the move
        :param objectives: The objectives obtained after applying the move
        :return: The modified solution
        """
        routes = solution.structure
        n = len(indexesModifiedRoutes)
        nb_to_remove = 0
        for i in range(n):
            index_route = indexesModifiedRoutes[i]
            new_route = newRoutes[i]
            routes[index_route] = new_route.copy()
            if new_route == [0,0]:
                nb_to_remove += 1
            else:
                self.problem.compute_subsequences(new_route[1:-1], solution, reverse = False)

        # Remove empty routes 
        for _ in range(nb_to_remove):
            routes.remove([0,0])

        # Update the list of variables
        new_variables = []
        for r in routes:
            for c in r:
                if c != 0:
                    new_variables.append(c-1)
        
        solution.variables = new_variables.copy()
        solution.objectives[0] = objectives[0]
        solution.objectives[1] = objectives[1]
        solution.improvedByLS = True
        return solution

class boVRPTWLocalSearch(GenericPermutationLocalSearch):
    def __init__(self, problem: MOVRPTW, neighbours, probability = 0, strategy = "") -> None:
        super(boVRPTWLocalSearch, self).__init__(problem, neighbours, probability, strategy)

    def evaluate_move(self, modifiedRoutes, solution):
        t1, t2 = modifiedRoutes[0], modifiedRoutes[1]
        newRoute1, attributesR1, attributesNewR1 = t1
        newRoute2, attributesR2, attributesNewR2 = t2
        
        if newRoute2 == None:
            former_cost = attributesR1['C']
            new_cost = attributesNewR1['C']

            new_wt = attributesNewR1['WT']
            former_wt = attributesR1['WT']
            
            check_delay = attributesNewR1['TW'] == 0
            check_capacity = attributesNewR1['Q'] <= self.problem.capacity
        else:
            former_cost = attributesR1['C'] + attributesR2['C']
            new_cost = attributesNewR1['C'] + attributesNewR2['C']

            former_wt = attributesR1['WT'] + attributesR2['WT']
            new_wt = attributesNewR1['WT'] + attributesNewR2['WT']
            
            check_delay = attributesNewR1['TW'] + attributesNewR2['TW'] == 0
            check_capacity = attributesNewR1['Q'] + attributesNewR2['Q'] <= self.problem.capacity 
        
        current_fitness = solution.attributes["weights"][0] * former_cost + solution.attributes["weights"][1] * former_wt
        new_fitness = solution.attributes["weights"][0] * new_cost + solution.attributes["weights"][1] * new_wt
        
        feasible = check_capacity and check_delay
        improved = round(new_fitness, 5) < round(current_fitness, 5)

        deltaObjectives = [new_cost - former_cost, new_wt-former_wt]
        return feasible, improved, deltaObjectives

class boTSPLocalSearch(GenericPermutationLocalSearch):
    def __init__(self, problem: BOTSP, neighbours, probability=0, strategy="") -> None:
        super(boTSPLocalSearch, self).__init__(problem= problem, neighbours= neighbours, probability= probability, strategy= strategy)
    
    def evaluate_move(self, modifiedRoutes, solution: RoutingSolution):
        new_route1, attributes_R1, attributes_newR1 = modifiedRoutes[0] # there is only one route

        former_cost1 = attributes_R1['cost1']
        former_cost2 = attributes_R1['cost2']

        new_cost1 = attributes_newR1['cost1']
        new_cost2 = attributes_newR1['cost2']

        current_fitness = solution.attributes["weights"][0] * former_cost1 + solution.attributes["weights"][1] * former_cost2
        new_fitness = solution.attributes["weights"][0] * new_cost1 + solution.attributes["weights"][1] * new_cost2
        
        improved = round(new_fitness, 5) < round(current_fitness, 5)

        deltaObjectives = [new_cost1 - former_cost1, new_cost2-former_cost2]
        return True, improved, deltaObjectives


