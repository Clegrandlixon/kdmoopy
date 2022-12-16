from abc import ABC, abstractmethod
import copy
import random

from jmetal.core.operator import Injection
from jmetal.core.solution import RoutingSolution
from jmetal.problem.multiobjective.movrptw import MOVRPTW
from jmetal.problem.multiobjective.motsp import BOTSP
from jmetal.operator.localSearch_vrptw import ApplyManyOperators

"""
.. module:: injection
   :platform: Unix, Windows
   :synopsis: Module implementing injection operators.

.. moduleauthor:: Clément Legrand <clement.legrand4.etu@univ-lille.fr>
"""

class GenericPatternInjection(Injection[RoutingSolution], ABC):
    def __init__(self, problem, probability: float, diversificationFactor: int, allowReverse: bool):
        super(GenericPatternInjection, self).__init__(probability, diversificationFactor)

        self.allowReverse = allowReverse
        self.problem = problem

        # Return True if at least one pattern has been injected in the solution
        self.atLeastOneinjected = False
        
        ### Variables used during best_reconnect procedure ###
        self.R_best = [] 
        self.bestCost = None
        self.recursion = 0
        ###
        
        # Attribute that depends on the solution
        self.names_and_weights = []
        
        
    @abstractmethod
    def names_of_attributes_and_weights(self, solution):
        """ Return tuples (name, weight) of characteristics required for the evaluation of a fragment of the solution 
        """
        pass

    @abstractmethod
    def compute_information_fragment(self, fragment, solution, reverse = False):
        """ Compute the attributes of a fragment. It can depend on the problem solved
        """
        pass

    def cost_of_fragment(self, fragment):
        cost = 0
        for (c,w) in self.names_weights:
            cost += w*fragment[c]
        return cost

    def cost_of_list_of_fragments(self, list_fragments):
        total_cost = 0
        for i in list_fragments:
            total_cost += self.cost_of_fragment(i[1])
        return total_cost

    def check_is_pattern_present(self, solution, pattern):
        l = len(pattern)
        adaptedPattern = [k-1 for k in pattern]
        list_routes = solution.structure
        start_pattern = adaptedPattern[0]
        is_present = False
        for r in list_routes:
            if start_pattern in list_routes:
                i = r.index(start_pattern)
                if i+l-1 < len(r):
                    is_present = adaptedPattern == r[i:i+l]
            break
        return is_present

    def __compute_information_pattern(self, solution, pattern):
        attributes_pattern = solution.sequences[pattern[0]][pattern[0]]
        current_pattern = [pattern[0]]
        n = len(pattern)
        for i in range(1, n):
            next_customer = pattern[i]
            attributes_pattern = self.problem.concatenate_subsequences((current_pattern[0], current_pattern[-1]), attributes_pattern, (next_customer, next_customer), solution.sequences[next_customer][next_customer])
            current_pattern.append(next_customer)
        return attributes_pattern

    def cut_into_fragments_depotVersion(self, solution, pattern):
        # the "depot" is included in the pattern (e.g. case when solving the tsp)
        route = solution.structure[0]
        R_init = [route.copy()] # routes which contain at least one customer of pattern
        R_beg = [] # fragments of routes which start with the depot
        R_mid = [] # fragments of routes which don't contain the depot
        R_end = [] # fragments of routes which end with the depot
        self.R_best = R_init.copy()

        if pattern[0] == 0 or pattern[-1] == 0:
            attributes_pattern = self.__compute_information_pattern(solution, pattern)
            if pattern[0] == 0:
                R_beg.append((pattern, attributes_pattern))
            else:
                R_end.append((pattern, attributes_pattern))

        else:
            i = pattern.index(0)
            subPatternEnd = pattern[:i+1]
            subPatternBeg = pattern[i:]

            attributes_end = self.__compute_information_pattern(solution, subPatternEnd)
            attributes_beg = self.__compute_information_pattern(solution, subPatternBeg)
            
            R_beg.append(subPatternBeg, attributes_beg)
            R_end.append(subPatternEnd, attributes_end)

            frag = []
            for c in route[1:-1]:
                if c in pattern:
                    if frag != []:
                        attributes_fragment = self.compute_information_fragment(frag, solution, False)
                        if len(frag) > 1:
                            if self.allowReverse:
                                reversed_fragment = self.compute_information_fragment(frag, solution, True)
                            else:
                                reversed_fragment = None
                        else:
                            reversed_fragment = attributes_fragment.copy()
                        R_mid.append((frag, attributes_fragment, reversed_fragment))
                    frag = []
                else:
                    frag.append(c)
        return [], R_beg, R_mid, R_end

    def cut_into_fragments_classicVersion(self, solution, pattern):
        list_routes = solution.structure
        R_inv = [] # routes which not contain any customers in the pattern and thus are not modified
        R_init = [] # routes which contain at least one customer of pattern
        R_beg = [] # fragments of routes which start with the depot
        R_mid = [] # fragments of routes which don't contain the depot
        R_end = [] # fragments of routes which end with the depot

        for r in list_routes:
            frag = []
            inv = True
            for c in r:
                if c in pattern:
                    inv = False
                    if frag != []:
                        # Try to optimize -> define a global variable to store computed frags with their costs
                        attributes_fragment = self.compute_information_fragment(frag, solution, False)
                        if frag[0] == 0:
                            R_beg.append((frag, attributes_fragment))
                            
                        else:
                            if self.allowReverse:
                                reversed_fragment = self.compute_information_fragment(frag, solution, True)
                            else:
                                reversed_fragment = None
                            R_mid.append((frag, attributes_fragment, reversed_fragment)) # reversed fragments are used only when self.allowReverse is True 
                        frag = []
                else:
                    frag.append(c)
            if inv:
                attributes_fragment = self.compute_information_fragment(r, solution)
                R_inv.append((r,attributes_fragment))
            else:
                attributes_fragment = self.compute_information_fragment(r, solution)
                R_init.append((r,attributes_fragment))
                if frag != []:
                    attributes_fragment = self.compute_information_fragment(frag, solution)
                    R_end.append((frag, attributes_fragment))

        attributes_pattern = self.__compute_information_pattern(solution, pattern)
        if self.allowReverse:
            reversed_pattern = pattern.reverse()
            attributes_reversed_pattern = self.__compute_information_pattern(solution, reversed_pattern)
        else:
            attributes_reversed_pattern = None
        R_mid.append((pattern, attributes_pattern, attributes_reversed_pattern))
        self.R_best = R_init.copy()

        return R_inv, R_beg, R_mid, R_end

    def cut_into_fragments(self, solution, pattern):
        if 0 in pattern:
            result = self.cut_into_fragments_depotVersion(solution, pattern)
        else:
            result = self.cut_into_fragments_classicVersion(solution, pattern)
        return result 

    def best_reconnect_PILS(self, solution: RoutingSolution, beg, mid, end, complete):
        """
        Apply the best_reconnect procedure from PILS
        reference: Florian Arnold et al. "PILS: Exploring high-order neighborhoods by pattern mining and injection" (2021)
        """
        (R_beg, cost_beg) = beg
        (R_mid, cost_mid) = mid
        (R_end, cost_end) = end
        (R_complete, cost_complete) = complete
        self.recursion += 1
        totalCost = cost_beg + cost_mid + cost_end + cost_complete

        if self.recursion > 10000: # Keep this condition to avoid memory problems when the number of pieces to reconnect is too high
            return

        if totalCost < self.bestCost:
            if len(R_beg) == 0:
                self.R_best = R_complete.copy()
                self.atLeastOneinjected = True
                self.bestCost = totalCost
            else:
                f_beg = random.sample(R_beg, 1)[0]
                cost_deleted = self.cost_of_list_of_fragments([f_beg])

                for f_mid in R_mid:
                    new_R_beg_normal = R_beg.copy()
                    new_R_beg_normal.remove(f_beg)
                    
                    new_R_beg_reverse = R_beg.copy()
                    new_R_beg_reverse.remove(f_beg)
                    
                    new_R_mid = R_mid.copy()
                    new_R_mid.remove(f_mid)

                    new_costMid = cost_mid - self.cost_of_list_of_fragments([f_mid]) 
                    new_mid = (new_R_mid, new_costMid)

                    new_frag_normal = f_beg[0] + f_mid[0] # the new fragment (starts with depot)
                    attributes_frag_normal = self.problem.concatenate_subsequences((f_beg[0][0], f_beg[0][-1]), f_beg[1], (f_mid[0][0], f_mid[0][-1]), f_mid[1])
                    
                    new_R_beg_normal.append((new_frag_normal, attributes_frag_normal))
                    cost_added_normal = self.cost_of_list_of_fragments([new_R_beg_normal[-1]])
                    new_costBeg_normal = cost_beg - cost_deleted + cost_added_normal
                    new_beg_normal = (new_R_beg_normal, new_costBeg_normal)
                    
                    new_end = (R_end.copy(), cost_end)
                    new_complete = (R_complete.copy(), cost_complete)

                    self.best_reconnect_PILS(solution, new_beg_normal, new_mid, new_end, new_complete)
                    
                    if self.allowReverse:
                        f_mid[0].reverse()
                        new_frag_reverted = f_beg[0] + f_mid[0]
                        attributes_frag_reverted = self.problem.concatenate_subsequences((f_beg[0][0], f_beg[0][-1]), f_beg[1], (f_mid[0][0], f_mid[0][-1]), f_mid[2])

                        new_R_beg_reverse.append((new_frag_reverted, attributes_frag_reverted))    
                        cost_added_reverse = self.cost_of_list_of_fragments([new_R_beg_reverse[-1]])
                        new_costBeg_reverse = cost_beg - cost_deleted + cost_added_reverse
                        new_beg_reverse = (new_R_beg_reverse, new_costBeg_reverse)
                        self.best_reconnect_PILS(solution, new_beg_reverse, new_mid, new_end, new_complete)
                    
                if len(R_beg) != 1 or len(R_mid) == 0:
                    for f_end in R_end:
                        new_R_beg = R_beg.copy()
                        new_R_beg.remove(f_beg)
                        new_R_end = R_end.copy()
                        new_R_end.remove(f_end)

                        new_costEnd = cost_end - self.cost_of_list_of_fragments([f_end])
                        new_end = (new_R_end.copy(), new_costEnd)
                        new_costBeg = cost_beg - cost_deleted
                        new_beg = (new_R_beg.copy(), new_costBeg)


                        new_frag = f_beg[0] + f_end[0] # the new fragment (starts with depot)
                        attributes_frag = self.problem.concatenate_subsequences((f_beg[0][0], f_beg[0][-1]), f_beg[1], (f_end[0][0], f_end[0][-1]), f_end[1])

                        new_R = R_complete.copy()
                        new_R.append((new_frag, attributes_frag))

                        new_costR = cost_complete + self.cost_of_list_of_fragments([new_R[-1]])
                        new_complete = (new_R.copy(), new_costR)

                        new_mid = (R_mid.copy(), cost_mid)

                        self.best_reconnect_PILS(solution, new_beg, new_mid, new_end, new_complete)

    def inject_one_pattern(self, solution: RoutingSolution, pattern):
        """
        Try to inject the pattern into the solution by using the injection of PILS method.
        Weights are used to compute the fitness of the solution
        """
        # verify that the pattern is not already present in the solution
        if not self.check_is_pattern_present(solution, pattern):

            # first cut the routes of the solution into fragments
            R_inv, R_beg, R_mid, R_end = self.cut_into_fragments(solution, pattern)
            random.shuffle(R_mid) # bring diversity during the reconnection if all possibilities can not be tested (memory issue)
            cost_INV = self.cost_of_list_of_fragments(R_inv)
            
            self.bestCost = solution.attributes["weights"][0] * solution.objectives[0] + solution.attributes["weights"][1] * solution.objectives[1] - cost_INV

            cost_beg = self.cost_of_list_of_fragments(R_beg)
            cost_mid = self.cost_of_list_of_fragments(R_mid)
            cost_end = self.cost_of_list_of_fragments(R_end)
            
            # apply best_reconnect to merge R_beg, R_mid and R_end
            self.recursion = 0
            self.best_reconnect_PILS(solution, (R_beg, cost_beg), (R_mid, cost_mid), (R_end, cost_end), ([], 0))
            
            # R_best contains the reconnected routes
            solution.objectives = [0,0]
            new_routes = []
            all_routes = R_inv + self.R_best
            s = []
            names_objectives = self.problem.get_names_objectives()

            # Update the sequences
            for (r, _) in self.R_best: # routes modified
                self.problem.compute_subsequences(r[1:-1], solution, False)
                if self.allowReverse:
                    reversed_r = r.copy()
                    reversed_r.reverse()
                    self.problem.compute_subsequences(reversed_r[1:-1], solution, True)
            
            for (i, att) in all_routes:
                if i != [0,0]:
                    for c in i:
                        if c != 0:
                            s.append(c-1)

                    solution.objectives[0] += att[names_objectives[0]]
                    solution.objectives[1] += att[names_objectives[1]]
                    new_routes.append(i.copy())

            solution.objectives[0] = round(solution.objectives[0], 1)
            solution.objectives[1] = round(solution.objectives[1], 1)
            solution.variables = s.copy()
            solution.structure = copy.deepcopy(new_routes)
        return solution

    def execute(self, solution: RoutingSolution, chosenPatterns: list) -> RoutingSolution:
        """ Tentatively inject patterns from chosenPatterns one by one.  
        A pattern is kept only in case of improvement.

        :param solution: The solution that undergoes the injection
        :param chosenPatterns: The list of patterns that are tentatively injected
        :return: A tuple (new solution, applied) containing the new solution and a boolean which is True when the injection has been applied 
        """
        
        self.atLeastOneinjected = False
        if random.random() > self.probability:
            return solution, False

        self.names_weights = self.names_of_attributes_and_weights(solution)
        for pattern in chosenPatterns:
            self.R_best = []
            self.bestCost = None
            self.recursion = 0
            solution = self.inject_one_pattern(solution, pattern)
        return solution, True

    def get_name(self):
        return 'Multi-Objective pattern injection'

class PatternInjectionMOTSP(GenericPatternInjection):
    def __init__(self, problem: BOTSP, probability, diversificationFactor):
        super(PatternInjectionMOTSP, self).__init__(problem, probability, diversificationFactor, allowReverse = True)

    def names_of_attributes_and_weights(self, solution: RoutingSolution):
        return [('cost1', solution.attributes["weights"][0]), ('cost2', solution.attributes["weights"][1])]

    def compute_information_fragment(self, fragment, solution, reverse: bool = True):
        """
        Compute the attributes of a fragment of solution (cost1 and cost2)
        The solution must have its sequences up to date

        :param fragment: The fragment that is evaluated (the depot is added)
        :param solution: The solution that contains the fragment
        :param reverse: Specify whether the fragment is reversed or not
        :return: A list containing the attributes of the fragment
        """
        if len(fragment) == 1:
            attributes_route = solution.sequences[fragment[0]][fragment[0]]
        elif fragment[0] == 0 and fragment[-1] == 0:
            attributes_route = solution.sequences[fragment[1]][fragment[-2]]
            attributes_route = self.problem.add_depot_to_sequence((fragment[1], fragment[-2]), attributes_route, solution)
        elif fragment[0] == 0:
            attributes_route = solution.sequences[fragment[1]][fragment[-1]]
            attributes_route = self.problem.add_depot_to_left((fragment[1], fragment[-1]), attributes_route, solution)
        elif fragment[-1] == 0:
            attributes_route = solution.sequences[fragment[0]][fragment[-2]]
            attributes_route = self.problem.add_depot_to_right((fragment[0], fragment[-2]), attributes_route, solution)
        else:
            if reverse:
                attributes_route = solution.reverted_sequences[fragment[0]][fragment[-1]]
            else:
                attributes_route = solution.sequences[fragment[0]][fragment[-1]]
        return attributes_route    

    def get_name(self):
        return 'Multi-Objective pattern injection for BOTSP'


class PatternInjectionMOVRPTW(GenericPatternInjection):
    def __init__(self, problem: MOVRPTW, probability, diversificationFactor):
        super(PatternInjectionMOVRPTW, self).__init__(problem, probability, diversificationFactor, allowReverse= False)

    def names_of_attributes_and_weights(self, solution: RoutingSolution):
        """
        Return the names of the characteristics necessary for the evaluation of a fragment of the solution 
        associated with their weights
        """
        return [('nQ', 5000), ('C', solution.attributes["weights"][0]), ('WT', solution.attributes["weights"][1]), ('TW', 50000)]

    def compute_information_fragment(self, fragment, solution: RoutingSolution, reverse = False):
        """
        Compute the attributes of a fragment of solution (duration, cost, waiting time...)
        The solution must have its sequences up to date

        :param fragment: The fragment that is evaluated (the depot is added)
        :param solution: The solution that contains the fragment
        :param reverse: Specify whether the fragment is reversed or not. It is set to False in the case of the movrptw
        :return: A list containing the attributes of the fragment
        """
        if len(fragment) == 1:
            attributes_route = solution.sequences[fragment[0]][fragment[0]]
        elif fragment[0] == 0 and fragment[-1] == 0:
            attributes_route = solution.sequences[fragment[1]][fragment[-2]]
            attributes_route = self.problem.add_depot_to_sequence((fragment[1], fragment[-2]), attributes_route, solution)
        elif fragment[0] == 0:
            attributes_route = solution.sequences[fragment[1]][fragment[-1]]
            attributes_route = self.problem.add_depot_to_left((fragment[1], fragment[-1]), attributes_route, solution)
        elif fragment[-1] == 0:
            attributes_route = solution.sequences[fragment[0]][fragment[-2]]
            attributes_route = self.problem.add_depot_to_right((fragment[0], fragment[-2]), attributes_route, solution)
        else:
            attributes_route = solution.sequences[fragment[0]][fragment[-1]]
        
        return attributes_route    

    def get_name(self):
        return 'Multi-Objective pattern injection for BOVRPTW'