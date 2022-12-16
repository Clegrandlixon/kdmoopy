import copy
import random
from typing import List
from collections import Counter


from jmetal.core.operator import Extraction
from jmetal.core.solution import RoutingSolution

"""
.. module:: extraction
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Clément Legrand <clement.legrand4.etu@univ-lille.fr>
"""
class PatternExtractionVRPTW(Extraction[RoutingSolution, list]):
    def __init__(self, diversificationFactor: int):
        super(PatternExtractionVRPTW, self).__init__(diversificationFactor)

    def extract_fixed_size_patterns_from_one_solution(self, solution: RoutingSolution, sizePattern: int) -> list:
        """ Extract patterns of a fixed size from a solution. 
        """

        list_of_tours = solution.structure
        all_patterns = []
        for tour in list_of_tours:
            patterns = []
            length = len(tour)
            for i in range(1,length-sizePattern): # do not consider patterns which contain the depot --> for routing problems ! (not the case for tsp)
                pattern = tour[i:i+sizePattern]
                patterns.append(pattern)
            all_patterns += patterns
        return all_patterns

    def execute(self, solution:RoutingSolution, maxPatternSize: int) -> list:

        all_patterns = []
        for sizePattern in range(2, maxPatternSize+1):
            patterns = self.extract_fixed_size_patterns_from_one_solution(solution, sizePattern)
            all_patterns += patterns
        return all_patterns

    def get_name(self):
        return 'Multi-Objective pattern extraction'

class PatternExtractionTSP(Extraction[RoutingSolution, list]):
    def __init__(self, diversificationFactor):
        super(PatternExtractionTSP, self).__init__(diversificationFactor)

    def extract_fixed_size_patterns_from_one_solution(self, solution: RoutingSolution, size_of_pattern: int) -> list:
        """ Extract patterns of a fixed size from a solution. 
        """

        list_of_tours = solution.structure
        all_patterns = []
        for tour in list_of_tours:
            patterns = []
            length = len(tour)
            for i in range(length):
                if i + size_of_pattern >= length:
                    pattern = tour[i:length] + tour[:i+size_of_pattern-length]
                else:
                    pattern = tour[i:i+size_of_pattern]
                patterns.append(pattern)
            all_patterns += patterns
        
        return all_patterns

    def execute(self, solution:RoutingSolution, maxPatternsSize: int) -> list:
        """ Extract the patterns with a size lower than maxPatternSize 
        """
        all_patterns = []
        for sizePattern in range(2, maxPatternsSize+1):
            patterns = self.extract_fixed_size_patterns_from_one_solution(solution, sizePattern)
            all_patterns += patterns
        return all_patterns

    def get_name(self):
        return 'Multi-Objective pattern extraction'