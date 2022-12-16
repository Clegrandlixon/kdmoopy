from concurrent.futures import process
import os
import subprocess
from multiprocessing import Pool

def call_script(args):
    print(args)
    algo, benchmark, run, number, typ, time, popSize, neighbours, nbgps = args
    subprocess.check_call(['./generate_jobs_RelationGroupsPop.sh', algo, benchmark, run, number, typ, time, popSize, neighbours, nbgps]) 

BENCHMARK = "Solomon"
ID_RUNS = [1, 30] # format [start, end] where the id of the first (resp. last) run is start (resp. end)
SIZES_INSTANCE = [100]
TIME_LIMIT = [720]
NUMBER_CORES = 8
NAME_ALGORITHMS = ["Base", "A-5Gps-I1-E1"] # see the generate_jobs file for the other names

if BENCHMARK == "Solomon":
    # Solomon
    numbers_instances_C = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '204', '205', '206', '207', '208']
    numbers_instances_R = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211']
    numbers_instances_RC = ['101', '102', '103', '104', '105', '106', '107', '108', '201', '202', '203', '204', '205', '206', '207', '208']
    
    number_instances = {}
    number_instances['C'] = numbers_instances_C
    number_instances['R'] = numbers_instances_R
    number_instances['RC'] = numbers_instances_RC

    type_instance_sol = ['R', 'C', 'RC']

elif BENCHMARK == "Generated":
    # Generated
    number_instances_gen = ['1', '2', '3', '4', '5', '6', '7', '8']
    type_instance_gen = []
    l1 = ['C', 'M', 'R']
    l2 = ['S', 'L']
    l3 = ['T', 'W']
    for a in l1:
        for b in l2:
            for c in l3:
                type_instance_gen.append(a+b+c)

allArgs = []
for algo in NAME_ALGORITHMS:
    for TYPE in type_instance_sol:
        for NUMBER in number_instances[TYPE]:
            for RUN in range(ID_RUNS[0], ID_RUNS[1]+1):
                for (s,t) in zip(SIZES_INSTANCE, TIME_LIMIT):
                    allArgs.append([algo, 'Solomon', str(RUN), NUMBER, TYPE, str(s), str(t)])
                        

with Pool(processes= NUMBER_CORES) as pool:
    pool.map(call_script, allArgs)
