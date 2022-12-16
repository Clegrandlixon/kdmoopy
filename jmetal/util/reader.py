import os
from jmetal.core.solution import PermutationSolution


# format Objectives: SNAPSHOT.FUN.k.tsv
# format Solutions: SNAPSHOT.VAR.k.tsv

def read_objectives(file):
    listObjectives = []
    with open(file, 'r') as of:
        lines = of.readlines()
        data = [line.lstrip() for line in lines if line !=""]
        for line in data:
            objectives = [float(i) for i in line.split()]
            listObjectives.append(objectives[1:])
    return listObjectives

def read_variables(file):
    listVariables = []
    with open(file, 'r') as of:
        lines = of.readlines()
        data = [line.lstrip() for line in lines if line !=""]
        for line in data:
            variables = [int(i) for i in line.split()]
            listVariables.append(variables)
    return listVariables


def read_checkpoint_k(path, k, problem):
    fileObjectives = os.path.join(path, 'SNAPSHOT.FUN.'+str(k)+'.tsv')
    listObjectives = read_objectives(fileObjectives)

    fileVariables = os.path.join(path, 'SNAPSHOT.VAR.'+str(k)+'.tsv')
    listVariables = read_variables(fileVariables)

    n = len(listObjectives)
    listSolutions = []

    for i in range(n):
        solution = PermutationSolution(problem.number_of_variables, problem.number_of_objectives)
        solution.objectives = listObjectives[i]
        solution.variables = listVariables[i]
        solution.weights = [0.8, 0.2]
        problem.evaluate(solution)
        tour = [0] + [i+1 for i in solution.variables]
        problem.compute_subsequences(tour, solution, reverse = False)
        listSolutions.append(solution)
    return listSolutions

def generate_setSolutions(listIndices, path, problem):
    setSolutions = []
    for k in listIndices:
        solutions = read_checkpoint_k(path, k, problem)
        setSolutions += solutions
    #return random.sample(setSolutions, 10)
    return setSolutions

def read_final_results(path, problem):
    """
    TODO: bTSP
    Read the front of a VRPTW instance. 
    """
    fileObjectives = os.path.join(path, 'FUN.tsv')
    listObjectives = read_objectives(fileObjectives)

    fileVariables = os.path.join(path, 'VAR.tsv')
    listVariables = read_variables(fileVariables)

    n = len(listObjectives)
    listSolutions = []
    cpt_false = 0
    for i in range(n):
        found = False
        solution = PermutationSolution(problem.number_of_variables, problem.number_of_objectives)
        #solution.objectives = listObjectives[i]
        solution.variables = listVariables[i]
        for j in range(0.0,1.1,0.1):
            solution.weights = [j, 1-j]
            problem.evaluate(solution)
            if solution.objectives == listObjectives[i]:
                found = True
                break
        if found:
            listSolutions.append(solution)
        else:
            cpt_false += 1
            solution.structure = [[0] + [i+1 for i in solution.variables] + [0]] # not represantative but contains all the patterns
            listSolutions.append(solution)
    print("Number of structures not found: ", cpt_false)
    return listSolutions

