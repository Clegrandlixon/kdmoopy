#from ast import pattern
import os
import sys
import getopt
import random
from threading import local
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from numpy.random.mtrand import rand
from jmetal.algorithm.multiobjective.learning import KnowledgeDiscoveryMOEAD
from jmetal.core.problem import Problem
from jmetal.util.aggregative_function import WeightedSum
from jmetal.util.termination_criterion import StoppingByTime
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.extraction import PatternExtractionVRPTW
from jmetal.operator.injection import PatternInjectionMOVRPTW
from jmetal.operator.mutation import PermutationSwapMutation, NullMutation
from jmetal.operator.localSearch_vrptw import ApplyManyOperators
from jmetal.problem.multiobjective.movrptw import MOVRPTW
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.core.solution import Solution
 
from jmetal.lab.visualization.plotting import Plot

SELECTION_PROBABILITY = 1.0
MAX_SIZE_DICTIONARY = 2000

def read_arguments(argv):
    try:                                
        opts, args = getopt.getopt(argv, "kf", ["printKnowledge", "useReferenceFront"])
    except getopt.GetoptError:                           
        sys.exit(2)  
    
    allParameters = {}
    nbArgs = len(argv[1:])

    # Default parameters
    allParameters["seed"] = 1
    allParameters["benchmark"] = "Solomon"
    allParameters["populationSize"] = 20
    allParameters["sizeNeighborhood"] = 5
    allParameters["probabilityCrossover"] = 0.50
    allParameters["probabilityMutation"] = 0.00
    allParameters["probabilityLocalSearch"] = 0.10
    allParameters["nbGroups"] = 5
    allParameters["maxPatternSize"] = 5
    allParameters["patternsInjected"] = 60
    allParameters["probabilityInjection"] = 1.00
    allParameters["displayKnowledge"] = False
    allParameters["useReferenceFront"] = False
    allParameters["lsStrategy"] = "FIRST-BEST"
    allParameters["metric"] = "d3"
    allParameters["injectionStrategy"] = 1
    allParameters["solutionsExtraction"] = "standard"
    allParameters["extractionStrategy"] = 1

    if opts:
        for opt in opts:
            
            if opt[0] == "-k":
                allParameters["displayKnowledge"] = True

            elif opt[0] == "-f":
                allParameters["useReferenceFront"] = True
            
    # Instance Parameters
    instanceParameters = ["nameAlgo", "seed", "benchmark", "numberOfRun", "instanceNumber", "instanceType", "instanceSize"]
    
    # Parameters related to MOEAD
    moeadParameters = ["maxTime","populationSize","sizeNeighborhood","metric","granularity","lsStrategy", 
        "probabilityCrossover","probabilityMutation","probabilityLocalSearch"]
    
    # Parameters related to the knowledge Discovery
    kdParameters = ["nbGroups","maxPatternSize","solutionsExtraction","extractionStrategy","patternsInjected","probabilityInjection","injectionStrategy"]

    orderParameters = instanceParameters + moeadParameters + kdParameters
    intParameters = ["seed","numberOfRun","maxTime","populationSize","sizeNeighborhood","granularity","nbGroups","maxPatternSize","patternsInjected","extractionStrategy","injectionStrategy"]
    floatParameters = ["probabilityCrossover","probabilityMutation", "probabilityLocalSearch","probabilityInjection"]
    
    for i in range(len(orderParameters)):
        nameParameter = orderParameters[i] 
        if nameParameter in intParameters:
            allParameters[nameParameter] = int(args[i])
        elif nameParameter in floatParameters:
            allParameters[nameParameter] = float(args[i])
        else:
            allParameters[nameParameter] = args[i]
    return allParameters

def read_front(path: str, nbObjectives: int = 2):
    """ Read the reference front from the file passed in argument.
    Each solution must have nbObjectives values.
    
    :param path: The path of the file containing the reference front
    :param nbObjectives: The number of objectives of a solution 
    :return: A list where each element represents the values of the objectives of a solution 
    """
    referenceFront = []
    with open(path) as file:
        lines = file.readlines()
        data = [line.lstrip() for line in lines if line != ""]
        for item in data:
            # you have to adapt this line if the file is formatted differently
            listObjectives = [float(x.strip()) for x in item[:-1].split(" ")][:nbObjectives]
            referenceFront.append(listObjectives)
    return referenceFront

def list_of_weights(nbWeights):
    """ Compute the weights used by MOEAD (bi-objective case).
    The weights are uniformly spread.
    
    :param nbWeights: The number of weights that need to be computed
    :return: The list of weights
    """
    x0, y0 = 0,1
    weights = [(x0, y0)]
    if nbWeights == 1:
        return weights
    for i in range(nbWeights-2):
        xi, yi = x0 + (i+1)/(nbWeights-1), y0 - (i+1)/(nbWeights-1)
        weights.append((xi, yi))
    weights.append((1,0))
    return weights

def neighbours_to_i(problem, i: int, metric: str = "d3", weight = None):
    """ Given a metric between customers, compute the distance from i to each customer

    :param problem: The problem considered (by default it is the bVRPTW)
    :param i: The index of the customer
    :param metric: The metric uses to compute the distances. It can be either "d1", "d2" or "d3". By default it should be "d3"
    :param weight: If the metric is "d2", you have to precise the weight given to each objective
    :return: The list of neighbours of i sorted by increasing distance
    """
    dimension = problem.number_of_variables + 1
    neighbours_i = [-1] * dimension
    for j in range(dimension): 
        obj1_ij = problem.metricsMatrices[0][i][j]
        nObj1_ij = (1 - (problem.extremaMetrics[0][1]-obj1_ij)/(problem.extremaMetrics[0][1] - problem.extremaMetrics[0][0]))
        
        obj2_ij = problem.metricsMatrices[1][i][j]
        nObj2_ij = (1 - (problem.extremaMetrics[1][1]-obj2_ij)/(problem.extremaMetrics[1][1] - problem.extremaMetrics[1][0]))
        
        if metric == "d1":
            fit_ij = nObj1_ij
        elif metric == "d2":
            (w1,w2) = weight
            fit_ij = w1 * nObj1_ij + w2 * nObj2_ij
        elif metric == "d3":
            fit_ij = nObj1_ij + nObj2_ij
        neighbours_i[j] = (fit_ij, j)
        
    neighbours_i.sort()
    return neighbours_i

def relevant_neighbours(nbSubproblems: int, problem, metric: str = "d3"):
    """ For each customer of the instance of the problem in argument, 
    compute its problem.granularity closest neighbours given a metric. 
    
    :param nbSubproblems: The number of subproblems generated in MOEAD
    :param problem: The problem solved by MOEAD
    :param metric: The metric uses to compute the distances. It can be either "d1", "d2" or "d3". By default it should be "d3"
    """
    reducedNeighbours = {} # the i-th list contains the neighborhood for the i-th subproblem
    weights = list_of_weights(nbSubproblems)
    for l in range(nbSubproblems):
        reducedNeighbours[l] = []
        dimension = problem.number_of_variables + 1
        weight = weights[l]
        for i in range(dimension):
            neighbours_i = neighbours_to_i(problem, i, metric, weight)
            isRelevantNeighbour_i = [False] * dimension
            for (_,j) in neighbours_i[:problem.granularity]:
                isRelevantNeighbour_i[j] = True
            reducedNeighbours[l].append(isRelevantNeighbour_i)
    return reducedNeighbours

def configure_experiment(allParameters: dict, subpath, path_front: str = None):
    """ This function creates the jobs that will be executed.
    It should be adapted according to the paths used, and the algorithm wanted.
    Here, it works with the file generate_jobsEMO2023
    """
    instance = os.path.join("resources", "VRPTW_instances", subpath)
    extremaPoints = os.path.join("resources", "extremaPoints_fronts", "VRPTW")
    if allParameters["benchmark"] == "Solomon":
        instance += ".txt"

    run = allParameters["numberOfRun"]
    initial_seed = allParameters["seed"]
    max_neighbours = allParameters["granularity"]

    if not path_front is None:
        referenceFront = read_front(path_front)
        problem = MOVRPTW(max_neighbours, instance, reference_front= referenceFront)
    else:
        extrema = os.path.join(extremaPoints, subpath, "extrema.txt")
        idealPoint = []
        nadirPoint = []
        if os.path.isfile(extrema):
            # If any, use the extrema available to generate an Ideal point and a Nadir point
            with open(extrema, "r") as file:
                lines = file.readlines()
                for line in lines:
                    values = [float(i) for i in line.split(" ")[:-1]]
                    idealPoint.append(values[0])
                    nadirPoint.append(values[1])

            problem = MOVRPTW(max_neighbours, instance, idealPoint= idealPoint, nadirPoint= nadirPoint)
        else:
            # In that case, the hypervolume is not computed during the execution
            problem = MOVRPTW(max_neighbours, instance)

    problem_tag = os.path.join(subpath, "Run"+str(run), "Final")
    reducedNeighbours = relevant_neighbours(allParameters["populationSize"], problem, allParameters["metric"])

    ls = ApplyManyOperators(problem, probability = allParameters["probabilityLocalSearch"], strategy= allParameters["lsStrategy"], neighbours= reducedNeighbours)
    mutation = NullMutation()

    job = Job(
            algorithm = KnowledgeDiscoveryMOEAD(
                problem = problem,
                aggregative_function = WeightedSum(),
                population_size = allParameters["populationSize"],
                optimizeInitialPopulation= False,
                neighbor_size = allParameters["sizeNeighborhood"],
                neighbourhood_selection_probability = SELECTION_PROBABILITY,
                max_number_of_replaced_solutions = 2,
                nbGroups= allParameters["nbGroups"],
                maxPatternSize= allParameters["maxPatternSize"],
                number_of_patterns_injected = allParameters["patternsInjected"],
                crossover = PMXCrossover(probability = allParameters["probabilityCrossover"]),
                mutation = mutation,
                extraction = PatternExtractionVRPTW(diversificationFactor= allParameters["extractionStrategy"]),
                solutionsForExtraction= allParameters["solutionsExtraction"],
                injection = PatternInjectionMOVRPTW(problem, probability = allParameters["probabilityInjection"], diversificationFactor= allParameters["injectionStrategy"]),
                localSearch= ls,
                weight_files_path = "resources/MOEAD_weights",
                output_path = os.path.join("data", allParameters["nameAlgo"], subpath, "Run"+str(run), "Snapshots"),
                extrema_path= os.path.join(extremaPoints, subpath),
                verbose= False,
                timeRecordData= -1,
                termination_criterion = StoppingByTime(allParameters["maxTime"])),
            algorithm_tag= allParameters["nameAlgo"],
            problem_tag=problem_tag,
            run=run,
            seed = initial_seed * (run)* 10
        )

    return job

#############################################################################################
# The following functions are used only when the knowledge is displayed after the execution #
#############################################################################################

def plot_instance(ax, problem: MOVRPTW):
    positions = problem.positions
    x = []
    y = []
    for i in range(len(positions)):
        if i%2 == 0:
            x.append(positions[i])
        else:
            y.append(positions[i])
    ax.scatter(x[0], y[0], color = "red")
    ax.scatter(x[1:], y[1:], color = "blue")
    return

def plot_patterns(ax, patterns, problem: MOVRPTW):
    # arcs contain the list of patterns to plot
    # a pattern is modeled as a list 
    positions = problem.positions
    frequency_max = patterns[0][1]
    for (pattern, frequency) in patterns:
        lgth = len(pattern)
        for i in range(lgth-1):
            a = pattern[i]
            b = pattern[i+1]
            xa = positions[2*a]
            ya = positions[2*a+1]
            xb = positions[2*b]
            yb = positions[2*b+1]
            ax.plot([xa, xb], [ya, yb], color = "black", linewidth = frequency/frequency_max) # could add a linewidth depending on the frequency of the pattern
    return

def computePatterns(group, sizePattern, numberPatterns, knowledge):
    currentKnowledge = knowledge[group][sizePattern]
    listPatterns = []
    sorted_knowledge = sorted(currentKnowledge.items(), key=lambda t: -t[1])
    relevant_patterns = sorted_knowledge[:numberPatterns]
    for element in relevant_patterns:
        pattern = [int(i) for i in element[0][1:-1].split(", ")[:-1]]
        listPatterns.append((pattern, element[1]))
    return listPatterns

def updateFigure(val):
    sizePattern = size_slider.val
    group = group_slider.val
    nbPatterns = nbPattern_slider.val
    listPatterns = computePatterns(group, sizePattern, nbPatterns, knowledgeGenerated)
    ax.clear()
    plot_instance(ax, problem)
    plot_patterns(ax, listPatterns, problem)
    return

#########################
# Core of the execution #
#########################

if __name__ == '__main__':
    # Read arguments
    allParameters = read_arguments(sys.argv[1:])

    # Configure the experiments for ONE instance
    instanceNumber = allParameters["instanceNumber"]
    instanceType = allParameters["instanceType"]
    instanceSize = allParameters["instanceSize"] 
    
    if allParameters["benchmark"] == "Solomon":
        subpath = os.path.join("Solomon", "solomon_"+instanceSize, instanceType + instanceNumber)
    
    elif allParameters["benchmark"] == "Generated":
        if int(instanceNumber) < 10:
            instanceNumber = "0" + instanceNumber
        subpath = os.path.join("generated_instances", "gen_"+instanceSize, instanceType + instanceNumber)

    if allParameters["useReferenceFront"]:
        path_front = os.path.join("resources", "reference_front", "VRPTW", allParameters["benchmark"], allParameters["benchmark"] + "_" + str(instanceSize), instanceType + instanceNumber, "FUN.tsv")
    else:
        path_front = None

    job = configure_experiment(allParameters, subpath, path_front)
        
    # Run the study
    output_directory = "data"
    experiment = Experiment(output_dir=output_directory, jobs=[job])
    experiment.run()

    output_path = os.path.join(output_directory, job.algorithm_tag, job.problem_tag)

    # If activated, display the content of the knowledge groups 
    if allParameters["displayKnowledge"]:
        knowledgeGenerated = job.algorithm.knowledge
        maxPatternSize = allParameters["maxPatternSize"]
        numberOfGroups = allParameters["nbGroups"]
        for i in range(0, numberOfGroups, 10):
            print("Aggregation of group ", i, ": ", job.algorithm.solutions[i].weights)
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom = 0.30)
    
        init_patternSize = 2
        init_group = 0
        init_nbPatterns = 10
        listPatterns = computePatterns(init_group, init_patternSize, init_nbPatterns, knowledgeGenerated)
        problem = job.algorithm.problem
        
        plot_instance(ax, problem)
        plot_patterns(ax, listPatterns, problem)

        axcolor = "White"
        
        # Set the axis
        group_axis = plt.axes([0.25, 0.2, 0.60, 0.03], 
                                facecolor = axcolor)
        patternSize_axis = plt.axes([0.25, 0.15, 0.60, 0.03], 
                                facecolor = axcolor)
        nbPattern_axis = plt.axes([0.25, 0.1, 0.60, 0.03], 
                                facecolor = axcolor)
        
        # Set the slider for the groups, the size of patterns and number of patterns
        group_slider = Slider(group_axis, "Group", 
                                0, numberOfGroups-1, valinit = init_group, valstep = 1)
        size_slider = Slider(patternSize_axis, "SizePatterns", 
                                2, maxPatternSize, valinit = init_patternSize, valstep = 1)
        nbPattern_slider = Slider(nbPattern_axis, "Nb Patterns", 
                                1, 50, valinit = init_nbPatterns, valstep = 1)

        group_slider.on_changed(updateFigure)
        size_slider.on_changed(updateFigure)
        nbPattern_slider.on_changed(updateFigure)
        
        plt.show()
    