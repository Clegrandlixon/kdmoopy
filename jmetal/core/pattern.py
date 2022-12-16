from abc import ABC
import random
from typing import List, Generic, TypeVar
import matplotlib.pyplot as plt
import numpy
from scipy import stats

from jmetal.util.ckecking import Check

BitSet = List[bool]
S = TypeVar('S')

MINIMUM_FREQUENCY = 0
MOST_FREQUENT_PATTERNS = 100
computeGap = lambda c1,c2: min(1, max(round(1 - (c2+0.1)/(c1+0.1), 3), -1))

class Pattern():
    """ Class representing patterns """

    def __init__(self, pattern: list, nbObjectives: int) -> None:
        self.name = str(pattern)
        self.size = len(pattern)
        self.pattern = pattern
        self.frequency = 1
        self.isfrequent = self.frequency >= MINIMUM_FREQUENCY
        self.profile = [[[] for _ in range(nbObjectives)] for _ in range(nbObjectives)]
        self.nbObjectives = nbObjectives
        self.averageImprovement = [[0 for _ in range(nbObjectives)] for _ in range(nbObjectives)]
    
    def get_name(self) -> str:
        return self.name

    def is_frequent(self):
        self.isfrequent = self.frequency >= MINIMUM_FREQUENCY
        return self.isfrequent

    def increment_frequency(self):
        self.frequency += 1

class StorePatterns():
    """ Class representing a data structure to store patterns """

    def __init__(self, maxPatternSize: int, nbObjectives: int, nbGroups):
        self.seenPatterns = [[] for _ in range(2, maxPatternSize+1)] # list of names of patterns seen
        self.unfrequentPatterns = [{} for _ in range(2,maxPatternSize+1)] # patterns that are not frequent
        self.frequentPatterns = [{} for _ in range(2, maxPatternSize+1)]   # patterns declared frequent
        self.nbObjectives = nbObjectives
        self.groups = [[set() for _ in range(2, maxPatternSize+1)] for _ in range(nbGroups)]

    def store_pattern(self, pattern: list, groups: list):
        name_pattern = str(pattern)
        size_pattern = len(pattern)
        if not (name_pattern in self.seenPatterns[size_pattern-2]):
            newPattern = Pattern(pattern, self.nbObjectives)
            if newPattern.is_frequent():
                self.frequentPatterns[size_pattern-2][name_pattern] = newPattern
                for idGroup in groups:
                    self.groups[idGroup][size_pattern-2].add(newPattern)
            else:
                self.unfrequentPatterns[name_pattern] = newPattern
        elif name_pattern in self.unfrequentPatterns[size_pattern-2].keys():
            storedPattern = self.unfrequentPatterns[name_pattern]
            storedPattern.increment_frequency()
            if storedPattern.is_frequent():
                self.unfrequentPatterns[size_pattern-2].pop(name_pattern)
                self.frequentPatterns[size_pattern-2][name_pattern] = storedPattern
                for idGroup in groups:
                    self.groups[idGroup][size_pattern-2].add(storedPattern)

        else:
            self.frequentPatterns[size_pattern-2][name_pattern].increment_frequency()

    def choosePatterns(self, groupID, sizePattern, nbChosen):
        patterns = self.groups[groupID][sizePattern-2]
        listOfPatterns = [p for p in patterns]
        listOfPatterns.sort(key=lambda t: -t.frequency)
        mostFrequentPatterns = listOfPatterns[:MOST_FREQUENT_PATTERNS]
        chosenPatterns = random.sample(mostFrequentPatterns, min(nbChosen, len(mostFrequentPatterns)))
        return [p.pattern for p in chosenPatterns]