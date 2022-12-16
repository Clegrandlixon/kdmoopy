import random as rd
import os

# benchmark KRO: uniform distribution in rectangle 4000*2000

def generate_city() -> tuple:
    return (rd.randint(0, 4000), rd.randint(0, 2000))

def generate_cities(total):
    cities = []
    for _ in range(total):
        cities.append(generate_city())
    return cities


def write_instance(instanceName, output, cities):
    size = len(cities)
    if not os.path.exists(output):
        os.makedirs(output)
    file_path = os.path.join(output, instanceName + '.tsp')
    with open(file_path, 'w') as file:
        file.write('NAME: ' + instanceName + '\n')
        file.write('TYPE: TSP\n')
        file.write('COMMENT: '+str(size)+'-city problem (LEGRAND) \n')
        file.write('DIMENSION: '+str(size)+'\n')
        file.write('EDGE_WEIGHT_TYPE: EUC_2D\n')
        file.write('NODE_COORD_SECTION\n')

        for i in range(size):
            city = cities[i]
            line = str(i+1) + " " + str(city[0]) + " " + str(city[1])  + "\n"
            file.write(line)
        file.write('EOF')
        
def generate_instances(total, size, seed):
    rd.seed(seed)
    for i in range(1, total+1):
        cities = generate_cities(size)
        instanceName = 'leg' + str(i)
        output_path = os.path.join("resources", 'TSP_instances', 'leg', 'leg_' + str(size))
        write_instance(instanceName, output_path, cities)
    return

generate_instances(9, 200, 1998)