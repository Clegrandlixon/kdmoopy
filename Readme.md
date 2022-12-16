## Project Overview
This project extends the jMetalPy framework (https://github.com/jMetal/jMetalPy).
If you encounter any problem related to the jMetalPy framework, please check before the link above.

This project is part of my PhD, and is consequently a *work in progress* project.
Once the paper associated is published (currently accepted at EMO2023), the reference will appear here. 

## Contact
If you have any suggestion/remarks or if you find a bug, do not hesitate to contact me at 
clement.legrand4.etu@univ-lille.fr.


## Features (added to the jMetalPy framework):

# Patterns
For now, patterns are the main characteristics that can be extracted from solutions. 
The module jmetal.core.pattern contains all important details concerning patterns.
Inside it, you can find a class Pattern, representing patterns themselves, and a class StorePatterns, that can be used to update and store patterns. 

# New operators
Two knowledge discovery operators have been added (for extraction and injection). 
Moreover localSearch operators for routing problems (vrptw) have been added too. 
These operators can be found in the module jmetal.operator

- Extraction ("extraction.py")
    * For now it can only extract patterns from solutions

- Injection ("injection.py")
    * It follows the injection procedure from PILS (F. Arnold), with some minor variations (e.g. reversed patterns)

-  Local Search ("localSearch_vrptw.py")
    * It combines three classical operators (Swap, Relocate and 2-opt*)
    * Two exploration strategies are available (either "First-Best" or "Best-Best")

# Miscellaneous
- There is now a class RoutingSolution, that is a specific case of PermutationSolution, and contains relevant information when solving routing problems (e.g. sequences, routes)

- The class Job (from the module jmetal.experiment) has been slightly modified to generate also a file containing statistics of the execution (like running times of functions and number of times it has been executed, but also the average improvement of operators)

## Description of Provided Files
The following files are also in the folder examples/knowledgeDiscovery

- "experiments_vrptw.py" shows how to use the knowledge discovery framework integrated into MOEA/D. 
It reads the parameters given and define an instance of the class KnowledgeDiscoveryMOEAD from jmetal algorithm.multiobjective.learning

- "generate_jobsEMO2023.sh" contains the configurations of the algorithms that are given to "experiments_vrptw.py"

- "manage_jobs.py" is used to run the experiments. It contains more generic information about the runs (e.g. name of the algorithms, the instances used, the execution time)

## Future Works:
 - Add a new problem: bi-objective TSP (most of the code is ready, but it needs to be polish first)


