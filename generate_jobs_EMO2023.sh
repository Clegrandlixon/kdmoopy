#!/bin/bash

#SBATCH --partition=2x24
#SBATCH --job-name=hostname
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

### Arguments to specify from left to right

# Option: -k (--printKnowledge): display the content of the knowledge groups after the execution
#    	  -f (--useReferenceFront): use the reference front stored in the file ""
# ($1) Name of the algorithm
# Seed
# ($2) Name of the Benchmark (either "Solomon" or "Generated")
# ($3) Id of the run
# ($4) Number of the instance (see the available instances)
# ($5) Type of instance (either C, R or RC for Solomon's instances)
# ($6) Size of the instance
# ($7) Termination criteron
# Size of the population (number of subproblems in MOEAD)
# Size of the neighborhood of a subproblem
# Metric used to measure the distance between two customers (either "d1", "d2" or "d3")
# Granularity (to prune the exploration space during the local search)
# Exploration strategy for the local search (either "FIRST-BEST" or "BEST-BEST")
# Probability of Crossover
# Probability of Mutation
# Probability of Local Search
# Number of Groups
# Maximum size of patterns extracted
# Solutions used for extraction (for now only "standard" is available)
# Extraction strategy
# Number of patterns injected
# Probability of injection
# Injection strategy

# Variants: format A-kGp-Im-En
# - kGps: use k knowledge groups
# - Im: the injection uses up to m groups (M refers to the population size)
# - En: the extraction updates up to n groups 

# Example:

# Run the following command to execute the algorithm A-5Gps-I1-E1 on the instance R101 of size 100 from Solomon benchmark with a budget of 720s

# sh generate_jobsEMO2023 A-5Gps-I1-E1 Solomon 1 101 R 100 720

# Check the file experiments_vrptw.py to see how the algorithm is built

case $1 in

    Base)	# equivalent to A-1Gp-I1-E1
	case $6 in
	    100)
		python3 experiments_vrptw.py -k $1 1 $2 $3 $4 $5 $6 $7 60 15 "d2" 50 "FIRST-BEST" 0.50 0.00 0.10 1 5 "standard" 1 60 0.75 1;;
	esac;;

    A-3Gps-I1-E1)
	case $6 in
	    100)
		python3 experiments_vrptw.py $1 1 $2 $3 $4 $5 $6 $7 60 15 "d2" 25 "FIRST-BEST" 0.50 0.00 0.10 3 5 "standard" 1 20 0.75 1;;
	esac;;
	
    A-3Gps-IM-E1)
	case $6 in
	    100)
		python3 experiments_vrptw.py $1 1 $2 $3 $4 $5 $6 $7 40 10 "d2" 25 "FIRST-BEST" 0.90 0.00 0.10 3 7 "standard" 1 40 1.00 3;;
	esac;;
	
    A-5Gps-I1-E1)
	case $6 in
	    100)
		python3 experiments_vrptw.py $1 1 $2 $3 $4 $5 $6 $7 40 10 "d2" 25 "FIRST-BEST" 0.50 0.00 0.25 5 7 "standard" 1 60 1.00 1;;
	esac;;


    A-5Gps-IM-E1)
	case $6 in
	    100)
		python3 experiments_vrptw.py $1 1 $2 $3 $4 $5 $6 $7 20 5 "d2" 25 "FIRST-BEST" 0.90 0.00 0.10 5 5 "standard" 1 60 0.90 5;;
	esac;;
		
    A-MGps-I1-E1)
	case $6 in
	    100)
		python3 experiments_vrptw.py $1 1 $2 $3 $4 $5 $6 $7 40 10 "d2" 25 "FIRST-BEST" 0.50 0.00 0.10 40 5 "standard" 1 40 1.00 40;;
	esac;;
	
    A-MGps-IM-E1)
	case $6 in
	    100)
		python3 experiments_vrptw.py $1 1 $2 $3 $4 $5 $6 $7 20 5 "d2" 25 "FIRST-BEST" 0.75 0.00 0.25 20 5 "standard" 1 40 0.90 20;;
	esac;;    
esac
