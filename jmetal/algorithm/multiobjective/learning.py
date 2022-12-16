import copy
import random
import os 
import time
from math import ceil, sqrt
from typing import TypeVar, List, Generator

from abc import abstractclassmethod, abstractmethod, ABC
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Selection, Mutation, Extraction, Injection, Crossover, GenericPermutationLocalSearch
from jmetal.core.problem import Problem, PermutationProblem
from jmetal.core.solution import RoutingSolution
from jmetal.core.pattern import StorePatterns
from jmetal.operator import NaryRandomSolutionSelection
from jmetal.util.aggregative_function import AggregativeFunction
from jmetal.util.evaluator import Evaluator
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.core.quality_indicator import HyperVolume, GenerationalDistance, InvertedGenerationalDistance, EpsilonIndicator
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.algorithm.multiobjective.moead import Permutation
S = TypeVar('S')
R = List[S]

class KnowledgeDiscoveryEvolutionaryAlgorithm(Algorithm[S, R], ABC):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int,
                 output_path: str, 
                 solutionsForExtraction: str):
        super(KnowledgeDiscoveryEvolutionaryAlgorithm, self).__init__(output_path)

        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.solutionsForExtraction = solutionsForExtraction

        self.knowledge = None
        self.groupNeighbors = []
        self.learningSet = [] # contains the set of solutions from which the knowedge is extracted
        self.memoryHorizon = 0 # count the number of iterations since the last reset

        # Snapshots and stats
        self.cpt = 1
        self.times = {"Injection": 0, "Extraction": 0, "Archive": 0, "Evaluation": 0, "LocalSearch": 0}
        self.countAll = {'Extraction': {'nothing': 0, 'ls': 0}, 'Injection': {'nothing': 0, 'crossover': 0}, 'LocalSearch': {'nothing': 0, 'crossover': 0, 'injection': 0, 'both': 0}}
        self.countImproved = {'Injection': {'nothing': 0, 'crossover': 0}, 'LocalSearch': {'nothing': 0, 'crossover': 0, 'injection': 0, 'both': 0}}
        self.crossoverApplied = False
        self.injectionApplied = False
        self.meanImprovement = {'Injection': {'nothing': 0, 'crossover': 0}, 'LocalSearch': {'nothing': 0, 'crossover': 0, 'injection': 0, 'both': 0}}
        self.metrics = {'Iteration': [], 'Time': [], 'SizeFront': [], 'unaryHypervolume': [], 'Hypervolume': [], 'Epsilon': [], 'GD': [], 'IGD': []}


    @abstractmethod
    def selection(self, population: List[S]) -> List[S]:
        """ Select the best-fit individuals for reproduction (parents). """
        pass

    @abstractmethod
    def reproduction(self, population: List[S]) -> List[S]:
        """ Breed new individuals through crossover and mutation operations to give birth to offspring. """
        pass

    @abstractmethod
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ Replace least-fit population with new individuals. """
        pass

    @abstractmethod
    def extractKnowledge(self, population: List[S]) -> list:
        """ Extract the knowledge from the current population and update the existing one. """
        pass

    @abstractmethod
    def injectKnowledge(self, population: List[S]) -> List[S]:
        """ Inject the knowledge inside some individuals from the population. """
        pass

    def get_observable_data(self) -> dict:
        front = self.get_result()
        if self.problem.ideal is None or len(front) == 0:
            epsilon = -1
            gd = -1
            igd = -1
            hv = 0.0
            uhv = 0.0
        elif self.problem.reference_front is None:
            uhv, hv = self.get_hypervolume(True), self.get_hypervolume(False)
            epsilon = -1
            gd = -1
            igd = -1
        else: 
            formattedFront = [i.objectives for i in front]
            uhv = self.get_hypervolume(True)
            hv = self.get_hypervolume(False)
            epsilon = EpsilonIndicator(self.problem.reference_front).compute(formattedFront)
            gd = GenerationalDistance(self.problem.reference_front).compute(formattedFront)
            igd = InvertedGenerationalDistance(self.problem.reference_front).compute(formattedFront)

        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': front,
                'unaryHYPERVOLUME': uhv,
                'HYPERVOLUME': hv,
                'EPSILON': epsilon,
                'GD': gd,
                'IGD':igd,
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def get_hypervolume(self, normalize: bool = False) -> float:
        """ Return the current hypervolume of the archive (stored in self.external_archive)
        """
        objectives = []
        if normalize:
            # Unary hypervolume
            indicator = HyperVolume([1.001, 1.001])
            for solution in self.external_archive:
                normalized = [round((solution.objectives[i] - self.problem.ideal[i])/max(1, (self.problem.nadir[i] - self.problem.ideal[i])),5) for i in range(len(solution.objectives))]
                objectives.append(normalized)
            hv = round(indicator.compute(objectives), 5)
        else:
            # Standard hypervolume
            indicator = HyperVolume([self.problem.nadir[0] + 0.1 * (self.problem.nadir[0] - self.problem.ideal[0]), self.problem.nadir[1] + 0.1 * (self.problem.nadir[1] - self.problem.ideal[1])])
            for solution in self.external_archive:
                objectives.append(solution.objectives)
            hv = round(indicator.compute(objectives), 0)
        return hv
        
        
    def init_progress(self) -> None:
        self.evaluations = self.population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def fast_insert(self, list: list, element):
        """ Dichotomic insertion """
        deb = 0
        fin = len(list)
        mid = (deb + fin)//2
        same = fin == 0
        found = False
        while not same:
            if list[mid].objectives > element.objectives:
                fin = mid
            elif list[mid].objectives < element.objectives:
                deb = mid
            else:
                found = True
                deb = mid
                fin = mid
            same = mid == (deb+fin)//2
            mid = (deb + fin)//2

        if not found:
            list.insert(fin, element)
        return list

    def __update_archive(self, solutions):
        for solution in solutions:
            self.fast_insert(self.external_archive, solution)

    def save_data(self):
        observables = self.get_observable_data()

        archive_time = time.time()
        self.external_archive = get_non_dominated_solutions(self.external_archive)
        self.times["Archive"] += time.time() - archive_time
        
        # Update the metrics
        self.metrics['Iteration'].append(self.iterations)
        self.metrics['Time'].append(round(observables['COMPUTING_TIME'], 2))
        self.metrics['SizeFront'].append(len(self.external_archive))
        self.metrics['Hypervolume'].append(round(observables['HYPERVOLUME'], 0))
        self.metrics['unaryHypervolume'].append(round(observables['unaryHYPERVOLUME'], 5))
        self.metrics['Epsilon'].append(int(observables['EPSILON']))
        self.metrics['GD'].append(round(observables['GD'], 2))
        self.metrics['IGD'].append(round(observables['IGD'], 2))

        # Store tatistics
        file_name = os.path.join(self.output_path, 'SNAPSHOT.FUN.{}.tsv'.format(self.cpt))
        print_function_values_to_file(self.external_archive, filename=file_name)

        file_name = os.path.join(self.output_path, 'SNAPSHOT.VAR.{}.tsv'.format(self.cpt))
        print_variables_to_file(self.external_archive, filename=file_name)

        file_name = os.path.join(self.output_path, 'SNAPSHOT.STATS.{}'.format(self.cpt))
        with open(file_name, 'w+') as of:
            of.write('Computing time: ' + str(observables['COMPUTING_TIME']) + '\n \n')
            of.write('Stats per bloc: ' + '\n' + '\n')
            of.write('Extraction: ' + '\n')
            of.write(' - Time: ' + str(self.times["Extraction"]) + '\n')
            of.write(' - Number Applied: ' + str(self.countAll['Extraction']) + '\n \n')
            of.write('Injection: ' + '\n')
            of.write(' - Time: ' + str(self.times["Injection"]) + '\n')
            of.write(' - When Crossover applied before: \n')
            of.write('  * Number Applied: ' + str(self.countAll['Injection']['crossover']) + '\n')
            of.write('  * Number Improved: ' + str(self.countImproved['Injection']['crossover']) + '\n')
            of.write('  * Mean Improvement: ' + str(round(self.meanImprovement['Injection']['crossover']/max(1, self.countAll['Injection']['crossover']), 3)) + '\n')
            of.write(' - When nothing applied before: \n')
            of.write('  * Number Applied: ' + str(self.countAll['Injection']['nothing']) + '\n')
            of.write('  * Number Improved: ' + str(self.countImproved['Injection']['nothing']) + '\n')
            of.write('  * Mean Improvement: ' + str(round(self.meanImprovement['Injection']['nothing']/max(1, self.countAll['Injection']['nothing']), 3)) + '\n \n')
            of.write('Evaluation: ' + '\n')
            of.write(' - Time: ' + str(self.times["Evaluation"]) + '\n')
            of.write(' - Number Applied: ' + str(self.evaluations) + '\n \n')
            of.write('Archive Update: ' + '\n')
            of.write(' - Time: ' + str(self.times["Archive"]) + '\n \n')
            of.write('Local Search: ' + '\n')
            of.write(' - Time: ' + str(self.times["LocalSearch"]) + '\n')
            of.write(' - When Crossover and Injection applied before: \n')
            of.write('  * Number Applied: ' + str(self.countAll['LocalSearch']['both']) + '\n')
            of.write('  * Number Improved: ' + str(self.countImproved['LocalSearch']['both']) + '\n')
            of.write('  * Mean Improvement: ' + str(round(self.meanImprovement['LocalSearch']['both']/max(1, self.countAll['LocalSearch']['both']), 3)) + '\n')
            of.write(' - When Injection applied before: \n')
            of.write('  * Number Applied: ' + str(self.countAll['LocalSearch']['injection']) + '\n')
            of.write('  * Number Improved: ' + str(self.countImproved['LocalSearch']['injection']) + '\n')
            of.write('  * Mean Improvement: ' + str(round(self.meanImprovement['LocalSearch']['injection']/max(1, self.countAll['LocalSearch']['injection']), 3)) + '\n')
            of.write(' - When Crossover applied before: \n')
            of.write('  * Number Applied: ' + str(self.countAll['LocalSearch']['crossover']) + '\n')
            of.write('  * Number Improved: ' + str(self.countImproved['LocalSearch']['crossover']) + '\n')
            of.write('  * Mean Improvement: ' + str(round(self.meanImprovement['LocalSearch']['crossover']/max(1, self.countAll['LocalSearch']['crossover']), 3)) + '\n')
            of.write(' - When nothing applied before: \n')
            of.write('  * Number Applied: ' + str(self.countAll['LocalSearch']['nothing']) + '\n')
            of.write('  * Number Improved: ' + str(self.countImproved['LocalSearch']['nothing']) + '\n')
            of.write('  * Mean Improvement: ' + str(round(self.meanImprovement['LocalSearch']['nothing']/max(1, self.countAll['LocalSearch']['nothing']), 3)) + '\n \n') 
            of.write('\n')
        self.cpt += 1

    def step(self, timeToRecord: int = -1):
        self.crossoverApplied = False
        self.injectionApplied = False
        self.lsApplied = False

        observables = self.get_observable_data()
        
        if (timeToRecord > 0) and observables['COMPUTING_TIME'] >= self.cpt * timeToRecord:
            self.save_data()
        
        # Select solutions
        mating_population = self.selection(self.solutions)
        
        # Crossover and then keep only one solution
        offspring_population = self.reproduction(mating_population)

        # Evaluate the offspring
        time_evaluate = time.time()
        offspring_population = self.evaluate(offspring_population)
        self.times["Evaluation"] += time.time() - time_evaluate

        # Inject the knowledge
        time_injection = time.time()
        offspring_population = self.injectKnowledge(offspring_population)
        self.times["Injection"] += time.time() - time_injection

        # Apply Local Search
        for s in offspring_population:
            time_ls = time.time()
            s = self.apply_local_search(s)
            self.times["LocalSearch"] += time.time() - time_ls
            
        self.learningSet += offspring_population

        # Update the neighborhood of the problem
        self.solutions = self.replacement(self.solutions, offspring_population)
        
        trueLearningSet = self.update_learningSet()

        archive_time = time.time()
        self.__update_archive(offspring_population)
        self.external_archive = get_non_dominated_solutions(self.external_archive)
        self.times["Archive"] += time.time() - archive_time
        
        # Extraction of knowledge
        time_extraction = time.time()
        self.knowledge = self.extractKnowledge(trueLearningSet)
        self.times["Extraction"] += time.time() - time_extraction

    def update_learningSet(self):
        trueSet = []
        if self.solutionsForExtraction == "":
            self.learningSet = []

        elif self.solutionsForExtraction == "standard" and self.iterations%self.population_size == 0:
            trueSet = self.learningSet.copy()
            self.learningSet = []
            
        return trueSet

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'


class KnowledgeDiscoveryGeneticAlgorithm(KnowledgeDiscoveryEvolutionaryAlgorithm[S]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 extraction: Extraction,
                 injection: Injection,
                 localSearch: GenericPermutationLocalSearch,
                 output_path: str,
                 solutionsForExtraction: str,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        super(KnowledgeDiscoveryGeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            output_path = output_path, 
            solutionsForExtraction = solutionsForExtraction)
            
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.injection_operator = injection
        self.extraction_operator = extraction
        self.ls_operator = localSearch

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.mating_pool_size = \
            self.offspring_population_size * \
            self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem)
                for _ in range(self.population_size)]

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break
            
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.extend(offspring_population)
        population.sort(key=lambda s: s.objectives[0])
        return population[:self.population_size]

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Genetic algorithm with Knowledge Discovery'

class KnowledgeDiscoveryMOEAD(KnowledgeDiscoveryGeneticAlgorithm):
    """ Algorithm adapted to problems with permutation solution, the mutation is the local search. """

    def __init__(self,
                 problem: PermutationProblem,
                 aggregative_function: AggregativeFunction,
                 weight_files_path: str,
                 population_size: int,
                 neighbor_size: int,
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 maxPatternSize: int,
                 crossover: Crossover,
                 mutation: Mutation, # It is possible to use any available mutation in the framework
                 localSearch: GenericPermutationLocalSearch,
                 nbGroups: int,
                 extraction: Extraction,
                 solutionsForExtraction: str,
                 injection: Injection,
                 number_of_patterns_injected: int,
                 output_path: str,
                 extrema_path: str,
                 optimizeInitialPopulation: bool = False,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 verbose: bool = False,
                 timeRecordData: int = -1): # if negative then the data is not saved during the execution, otherwise it is saved every timeRecordData seconds 
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
               than the entire population (Delta in Zhang & Li paper).
        """
        super(KnowledgeDiscoveryMOEAD, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            selection=NaryRandomSolutionSelection(2),
            crossover= crossover,
            mutation= mutation,
            extraction = extraction,
            injection = injection,
            localSearch= localSearch,
            output_path = output_path,
            solutionsForExtraction = solutionsForExtraction,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion
            )

        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions
        self.fitness_function = aggregative_function
        self.extrema_path = extrema_path
        self.neighbourhood = WeightVectorNeighborhood(
            number_of_weight_vectors=population_size,
            neighborhood_size=neighbor_size,
            weight_vector_size=problem.number_of_objectives,
            weights_path=weight_files_path
        )
        self.neighbourhood_selection_probability = neighbourhood_selection_probability
        self.permutation = None
        self.current_subproblem = 0
        self.neighbor_type = None
        self.adaptedNeighbors = {}
        self.maxPatternSize = maxPatternSize
        self.number_of_patterns_injected = number_of_patterns_injected
        self.nbGroups = nbGroups
        self.optimizeInitialPopulation = optimizeInitialPopulation
        self.associatedGroups = {}
        self.groups = {}
        self.knowledge = StorePatterns(self.maxPatternSize, self.problem.number_of_objectives, self.nbGroups)
        self.lastOperatorApplied = None 
        self.evaluateAfterOperators = True
        self.verbose = verbose
        self.timeRecordData = timeRecordData

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()
        
        # Initialisation Phase
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)
        self.initialize_groups()
        self.associate_problems_to_groups()        
        self.init_progress()

        # Optimising the Initial Population 
        if self.optimizeInitialPopulation:
            self.optimizePopulation()

        self.external_archive = self.solutions
        self.external_archive.sort(key= lambda s: s.objectives)

        # Main Loop 
        while (not self.stopping_condition_is_met()):
            if self.verbose:
                print("##################")
                print("Starting Iteration: ", self.iterations)
                print("Problem: ", self.current_subproblem, self.neighbourhood.weight_vectors[self.current_subproblem])
                print("Solution: ", self.solutions[self.current_subproblem].objectives, " Size set: ", len(self.learningSet))
                print()
            
            self.iterations += 1
            self.step(self.timeRecordData)
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time
        
    def optimizePopulation(self):
        for i in range(self.population_size):
            if not self.stopping_condition_is_met():
                if self.verbose:
                    print("Optimizing solution ", i, " out of ", self.population_size)
                s = self.solutions[i]
                time_LS = time.time()
                s = self.apply_local_search(s)
                self.LS_time += time.time() - time_LS
                self.update_progress()

    def initialize_groups(self):
        """ Associate a weight to each group
        """
        x0, y0 = 0,1
        cpt = 0
        self.groups[cpt] = (x0, y0)
        if self.nbGroups == 1:
            return
        for i in range(self.nbGroups-2):
            cpt += 1
            xi, yi = x0 + (i+1)/(self.nbGroups-1), y0 - (i+1)/(self.nbGroups-1)
            self.groups[cpt] = (xi, yi)
        self.groups[cpt+1] = (1,0)
        return 

    def associate_problems_to_groups(self):
        """ Define the region of each group by assigning a set of problems to each group
        In practice, self.associatedGroups[i] contains the indexes of the groups to update when finding a solution for the i-th subproblem
        """
        distances_PbToGp = [[-1] * self.nbGroups for _ in range(self.population_size)]
        for i in range(self.population_size):
            weights_pbi = self.neighbourhood.weight_vectors[i]
            for j in range(self.nbGroups):
                weights_gpj = self.groups[j]
                distances_PbToGp[i][j] = (sqrt((weights_pbi[0]-weights_gpj[0])**2 + ((weights_pbi[1]-weights_gpj[1])**2)), j)
        m = self.extraction_operator.diversificationFactor
        for i in range(self.population_size):
            distances_i = distances_PbToGp[i]
            distances_i.sort()
            groupsToUpdate = [j[1] for j in distances_i][:m]
            self.associatedGroups[i] = groupsToUpdate.copy()
        return
    
    def create_initial_solutions(self) -> List[S]:
        """ Create a set of solutions using the generator provided in the class of the problem.
        Then define the attributes of each new solution and compute the corresponding sequences (specific to routing).

        :return: A set of solutions (one for each subproblem)
        """
        solutions = []
        for i in range(self.population_size):
            new_solution = self.population_generator.new(self.problem)
            new_solution.attributes["weights"] = self.neighbourhood.weight_vectors[i]
            new_solution.associatedProblem = i
            tour = self.problem.get_tour(new_solution)
            self.update_sequences(tour, new_solution)
            solutions.append(new_solution)
        return solutions

    def apply_local_search(self, solution: RoutingSolution) -> RoutingSolution:
        """ Apply the local search provided (if any) to MOEAD.
        Update data concerning the mean improvement gap of the operator (for analysis only).
        
        :param solution: The solution that undergoes the local search
        :return: The solution obtained after the LS if applied (otherwise the initial solution)
        """
        if self.ls_operator.probability == 0:
            return solution
            
        fitnessBefore = 1 + solution.attributes["weights"][0] * solution.objectives[0] + solution.attributes["weights"][1] * solution.objectives[1]
        
        tour = self.problem.get_tour(solution)
        self.update_sequences(tour, solution)

        time_LS = time.time()
        solution = self.ls_operator.execute(solution)
        time_LS = time.time() - time_LS
        
        if self.evaluateAfterOperators:
            structure_before = solution.structure
            solution = self.problem.evaluate(solution)
            modified_routes = self.compare_structures(structure_before, solution.structure)
            for tour in modified_routes:
                self.update_sequences(self.problem.formating_tour(tour), solution)

        # Data for Local Search #
        fitnessAfter = 1 + solution.attributes["weights"][0] * solution.objectives[0] + solution.attributes["weights"][1] * solution.objectives[1]
        propImprovement = round(100*(fitnessBefore - fitnessAfter)/fitnessBefore, 3)

        if self.crossoverApplied and self.injectionApplied:
            self.meanImprovement['LocalSearch']['both'] += propImprovement
            self.countAll['LocalSearch']['both'] += 1
            self.countImproved['LocalSearch']['both'] += 1 if propImprovement > 0 else 0
        elif self.crossoverApplied:
            self.meanImprovement['LocalSearch']['crossover'] += propImprovement
            self.countAll['LocalSearch']['crossover'] += 1
            self.countImproved['LocalSearch']['crossover'] += 1 if propImprovement > 0 else 0
        elif self.injectionApplied:
            self.meanImprovement['LocalSearch']['injection'] += propImprovement
            self.countAll['LocalSearch']['injection'] += 1
            self.countImproved['LocalSearch']['injection'] += 1 if propImprovement > 0 else 0
        else:
            self.meanImprovement['LocalSearch']['nothing'] += propImprovement
            self.countAll['LocalSearch']['nothing'] += 1
            self.countImproved['LocalSearch']['nothing'] += 1 if propImprovement > 0 else 0
        ##########################
        return solution
        
    def init_progress(self) -> None:
        self.evaluations = self.population_size
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)
        self.permutation = Permutation(self.population_size)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def selection(self, population: List[S]):
        self.current_subproblem = self.permutation.get_next_value()
        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            mating_population = self.selection_operator.execute(population)

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        """ Apply a crossover operator to the mating_population in argument.
        If the crossover is not applied, then the solution curretnly associated to the problem is returned.
        Then a mutation operator occurs, and the attributes of the new solution are updated accordingly.

        :param mating_population: A list containing the solutions for the crossover
        :return: The solution obtained after the crossover and the mutation
        """
        # perform the crossover
        offspring_population, applied = self.crossover_operator.execute(mating_population)
        offspring_population = [offspring_population[random.randint(0,1)]]
        self.crossoverApplied = applied
        if not applied:
            # Then we keep the solution associated to the current subproblem
            offspring_population = [copy.copy(self.solutions[self.current_subproblem])]
        
        current_cycle = int((self.iterations-1)//self.population_size) + 1
        for solution in offspring_population:
            # perform the mutation
            solution = self.mutation_operator.execute(solution)
            weight_vector = self.neighbourhood.weight_vectors[self.current_subproblem]
            solution.attributes["weights"] = weight_vector
            solution.associatedProblem = self.current_subproblem
            solution.cycle = current_cycle
            if applied:
                tour = self.problem.get_tour(solution)
                self.update_sequences(tour, solution)
        return offspring_population

    def extractKnowledge(self, learningSet: List[RoutingSolution]) -> list:
        """ Extract the knowledge from the learningSet provided.
        Use the extraction_operator provided.
        
        :param learningSet: The set of solutions from which the knowledge will be extracted
        :return: The updated knowledge (it uses the structure of the class StorePatterns)
        """
        for solution in learningSet:
            patterns = self.extraction_operator.execute(solution= solution, maxPatternSize= self.maxPatternSize)
            if patterns != []:
                groupsToUpdate = self.associatedGroups[solution.associatedProblem]
                for pattern in patterns:
                    self.knowledge.store_pattern(pattern, groupsToUpdate)
        return self.knowledge

    def injectKnowledge(self, offspring_population: List[RoutingSolution]) -> List[RoutingSolution]:
        """ Inject the knowledge into the solutions given in parameter.
        
        :param offspring_population:
        :return: The list of solutions obtained after injection
        """
        if self.injection_operator.probability == 0:
            return offspring_population

        newSolutions = []
        for solution in offspring_population:
            tour = self.problem.get_tour(solution)
            self.update_sequences(tour, solution)
            possibleGiver = self.associatedGroups[self.current_subproblem][:self.injection_operator.diversificationFactor]
            indexReceiver = self.associatedGroups[self.current_subproblem][0]
            indexGiver = random.sample(possibleGiver, 1)[0]
            sizePattern = random.randint(2, self.maxPatternSize)
            patternsToInject = self.knowledge.choosePatterns(indexGiver, sizePattern, self.number_of_patterns_injected)

            fitness_before = self.fitness_function.compute(solution.objectives, solution.attributes["weights"])

            newSolution, applied = self.injection_operator.execute(solution, patternsToInject)
            
            if self.evaluateAfterOperators:
                structure_before = solution.structure
                solution = self.problem.evaluate(solution)
                modified_routes = self.compare_structures(structure_before, solution.structure)
                for tour in modified_routes:
                    self.update_sequences(self.problem.formating_tour(tour), solution)
            newSolutions.append(newSolution)
            fitness_after = self.fitness_function.compute(solution.objectives, solution.attributes["weights"])

            ### Improvement data ###
            gap = 1 - (fitness_after + 1)/(fitness_before + 1)
            propImprovement = round(100*gap, 3)
            self.injectionApplied = applied
            if self.crossoverApplied:
                self.meanImprovement['Injection']['crossover'] += propImprovement
                self.countAll['Injection']['crossover'] += 1
                self.countImproved['Injection']['crossover'] += 1 if propImprovement > 0 else 0
            else:
                self.meanImprovement['Injection']['nothing'] += propImprovement
                self.countAll['Injection']['nothing'] += 1
                self.countImproved['Injection']['nothing'] += 1 if propImprovement > 0 else 0
            
        return newSolutions

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ Update the population with the solutions found.
        """
        new_solution = offspring_population[0]
        self.fitness_function.update(new_solution.objectives)
        new_population = self.update_current_subproblem_neighborhood(new_solution, population)
        return new_population

    def update_current_subproblem_neighborhood(self, new_solution, population):
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(self.current_subproblem)
        replacements = 0

        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]
            weights_k = self.neighbourhood.weight_vectors[k]
            f1 = self.fitness_function.compute(population[k].objectives, weights_k)
            f2 = self.fitness_function.compute(new_solution.objectives, weights_k)
            if f2 < f1 or (f2 == f1 and (new_solution.objectives[0] < population[k].objectives[0] or new_solution.objectives[1] < population[k].objectives[1])):
               
                population[k] = copy.copy(new_solution) 
                replacements += 1
                population[k].objectives[0] = round(population[k].objectives[0], 2)
                population[k].objectives[1] = round(population[k].objectives[1], 2)
                population[k].weights = weights_k.copy()
                self.fitness_function.update(new_solution.objectives)   # update ideal point
            if replacements >= self.max_number_of_replaced_solutions:
                break

        return population

    def compare_structures(self, structureBefore, structureAfter):
        modified_routes = []
        for route in structureAfter:
            if not route in structureBefore:
                modified_routes.append(route)
        return modified_routes

    def update_sequences(self, tour, solution: RoutingSolution):
        """ Compute the sequences of a list of customers, which is contained in the solution provided in argument. 
        (Specific to routing solutions where sequences are used to accelerate the evaluation of neighbouring solutions).
        """
        self.problem.compute_subsequences(tour, solution, reverse = False)
        reverted_tour = tour.copy()
        reverted_tour.reverse()
        self.problem.compute_subsequences(reverted_tour, solution, reverse = True)
        return

    def generate_permutation_of_neighbors(self, subproblem_id):
        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighborhood()[subproblem_id]
            permuted_array = copy.deepcopy(neighbors.tolist())
        else:
            permuted_array = Permutation(self.population_size).get_permutation()

        return permuted_array

    def choose_neighbor_type(self):
        rnd = random.random()

        if rnd < self.neighbourhood_selection_probability:
            neighbor_type = 'NEIGHBOR'
        else:
            neighbor_type = 'POPULATION'

        return neighbor_type

    def get_name(self):
        return 'Knowledge Discovery MOEAD'

    def get_result(self):
        return get_non_dominated_solutions(self.external_archive)
