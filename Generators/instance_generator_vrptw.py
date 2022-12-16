## Legrand_generator
## It generates instances for VRPTW
## We use the same rules as described in:
## ~E. Uchoa et al., 2016, New benchmark instances for the Capacitated Vehicle Routing Problem

# Parameters
# NB: for all instances Customers are placed in a grid [0, 1000] * [0, 1000]
# NNB: it is also possible to consider a smaller grid [0, 100] * [0, 100] by dividing all parameters/bounds by 10 (except demands)
# (It reduces problems with floating approximation)

########################
# 1. Depot positioning #
########################
# Three possibilities:
#   C: depot = [500, 500] or [50, 50]
#   E: eccentric = [0, 0]
#   R: random (uniform distribution) 

############################
# 2. Customer positionning #
############################
# Three possibilities:
#   R: random (uniform distribution)
#   C: clustered
#       - Pick S in UD[3,8] (nb of clusters)
#       - Place these S customers randomly in the grid
#       - Seeds will attract N-S customers with exponential decay (cf article)
#       - Method: pick one point on the grid, keep it if proba respected otherwise pick another point
#   RC: random-clustered (50/50 of each method)

##########################
# 3. Demand Distribution #
##########################
# We keep only 3 possibilities:
#   Large values, large variation: demands from UD[1, 100]
#   Small values, small variation: demands from UD[5, 10] 
#   Many small, few large: most demands (70-95% of customers) are taken from UD[1, 10], the others from UD[50, 100]

#####################
# 4. Route Capacity #
#####################
# Pick r, a number which represents the longer of routes and apply the formula. 
# for example r in UD[N/25, N/4] (which allows a number of routes between 4 and 25)

###################
# 5. Time Windows #
###################
# First choose a TW for the depot
#   Large Horizon: UD[1000, 2500] or UD[100, 250]
#   Small Horizon: UD[500, 1000] or UD[50, 100]
# 
# Then TW for each customer:
#   Center of the TW:
#       - UD[0, depotHorizon]
#   Wide: width in UD[100, 500] or UD[25, 50]
#   Tight:width in UD[50, 250] or UD[5, 25]


###################
# 6. Service Time #
###################
# NB: time between two customers will be the euclidean distance  between them
# same possibilities as demand distributions:
#   Large values, large variation: service from UD[100, 200] or UD[50, 100]
#   small values, small variation: service from UD[50, 100] or UD[5, 10]

import os
import random as rd
import math as mth

# each point posseses 7 values when defined: id, x_coord, y_coord, demand, ready_time, due_date, service_time
# point with id: 0 is the depot

def create_depot(position, horizon):
    """
    Creates attributes for the depot according to the parameters given
    """
    # initialize the structure
    depot = {}
    depot['id'] = 0

    # place the depot
    if position == "Center":
        depot['x'] = 500
        depot['y'] = 500
    elif position == "Eccentric":
        depot['x'] = 0
        depot['y'] = 0
    elif position == "Random":
        depot['x'] = rd.randint(0, 1000)
        depot['y'] = rd.randint(0, 1000)
    else:
        raise ValueError('position can only be "Center", "Eccentric" or "Random"')
    depot['demand'] = 0
    depot['ready'] = 0

    # time horizon
    if horizon == "Large":
        depot['due'] = rd.randint(2500, 5000)
    elif horizon == "Small":
        depot['due'] = rd.randint(1500, 2500)
    else:
        raise ValueError('horizon can only be "Large" or "Small"')
    
    depot['service'] = 0
    return depot

def random_position(lb, ub) -> tuple:
    return (rd.randint(lb, ub), rd.randint(lb, ub))

def cluster_position(clusters: list, lb, ub) -> tuple:
    """
    return a valid position according to clusters
    clusters is a list of tuples, each tuple being the coordinated of a cluster
    """
    found = False
    while not found:
        position = random_position(lb, ub)
        proba_being_accepted = 0
        for i in range(len(clusters)):
            cluster_i = clusters[i]
            proba_being_accepted += mth.exp(-mth.dist(position, cluster_i) / 40)
        choice = rd.random()
        if choice <= proba_being_accepted:
            found = True
    return position

def demand_customer(demandDistribution):
    if demandDistribution == "Large":
        demand = rd.randint(1, 100)
    elif demandDistribution == "Medium":
        proba_small = rd.random()
        if proba_small <= 0.8:
            demand = rd.randint(1, 10)
        else:
            demand = rd.randint(50, 100)
    elif demandDistribution == "Small":
        demand = rd.randint(5, 10)
    else:
        raise ValueError('demandDistribution can only be "Large", "Medium" or "Small"')
    
    return demand 

def window_customer(horizonMin, horizonMax, widthTW):
    """
    return a time window for a customer
    need the horizon of depot, and the time between the customer and the depot 
    """
    center = rd.randint(0, horizonMax)

    if widthTW == "Wide":
        width = rd.randint(200, 500)
    elif widthTW == "Tight":
        width = rd.randint(50, 200)
    else:
        raise ValueError('widthTW can only be "Wide" or "Tight"')
    
    dueDate =  max(min(center + width//2, horizonMax), int(horizonMin * 1.1))
    readyDate = dueDate - width

    return [readyDate, dueDate] # ensure that the tw does not finish before the minimal time needed to go to him

def service_customer(serviceDistribution):
    if serviceDistribution == "Large":
        service = rd.randint(100, 200)
    elif serviceDistribution == "Small":
        service = rd.randint(50, 100)
    return service

def compute_attributes_customer(depotPosition, customer, position, demandDistribution, horizonMax, widthTW, serviceDistribution):
    customer['x'] = position[0]
    customer['y'] = position[1]
    customer['demand'] = demand_customer(demandDistribution)
    horizonMin = mth.sqrt((depotPosition[0]-position[0])**2 + (depotPosition[1]-position[1])**2)
    window = window_customer(horizonMin, horizonMax, widthTW)
    customer['ready'] = window[0]
    customer['due'] = window[1]
    customer['service'] = service_customer(serviceDistribution)

def create_customers(number, depotPosition, positionType, demandDistribution, horizon, widthTW, serviceDistribution):
    """
    creates the set of customers
    """
    all_customers = []
    if positionType == "Random":
        for id in range(1, number + 1):
            customer = {}
            customer['id'] = id
            positionCustomer = random_position(0, 1000)
            compute_attributes_customer(depotPosition, customer, positionCustomer, demandDistribution, horizon, widthTW, serviceDistribution)
            all_customers.append(customer)

    elif positionType == "Cluster":
        S = rd.randint(3, 8)
        clusters = []
        for id in range(1, number + 1):
            if id <= S:
                # then the customer is the seed of a cluster
                customer = {}
                customer['id'] = id
                positionCustomer = random_position(0, 1000)
                clusters.append(positionCustomer)
                compute_attributes_customer(depotPosition, customer, positionCustomer, demandDistribution, horizon, widthTW, serviceDistribution)
                all_customers.append(customer)
            
            else:
                customer = {}
                customer['id'] = id
                positionCustomer = cluster_position(clusters, 0, 1000)
                compute_attributes_customer(depotPosition, customer, positionCustomer, demandDistribution, horizon, widthTW, serviceDistribution)
                all_customers.append(customer)
            
    elif positionType == "Mixt":
        # first we create the cluster
        all_customers_cluster = create_customers(number//2, depotPosition, "Cluster", demandDistribution, horizon, widthTW, serviceDistribution)
        all_customers_random = create_customers(number - number//2, depotPosition, "Random", demandDistribution, horizon, widthTW, serviceDistribution)
        all_customers = all_customers_cluster + all_customers_random
        # need to update the ids of customers
        for i in range(number):
            all_customers[i]['id'] = i + 1

    return all_customers

def define_capacity(size, customers):
    r = rd.randint(size//15, size//5) # to adapt if needed
    total_demand = 0
    max_demand = 0
    for customer in customers:
        total_demand += customer['demand']
        if customer['demand'] > max_demand:
            max_demand = customer['demand']
    Q = max(mth.ceil(r * total_demand / size), int(max_demand * 1.1))

    return size//r, Q

def generate_instance(size, positionDepot, horizon, positionCustomer, demandDistribution, widthTW, serviceDistribution):
    depot = create_depot(positionDepot, horizon)
    horizonDepot = depot['due']
    customers = create_customers(size, [depot['x'], depot['y']], positionCustomer, demandDistribution, horizonDepot, widthTW, serviceDistribution)
    all_nodes = [depot] + customers
    vehicles, capacity = define_capacity(size, customers)
    return vehicles, capacity, all_nodes

def write_instance(instanceName, entete, output, nbVehicles, capacity, nodes):
    with open(output, 'w') as file:
        file.write(instanceName + '\n')
        file.write(entete)
        file.write('\n')
        file.write('VEHICLE \n')
        file.write('NUMBER \t CAPACITY \n')
        file.write(' ' + str(nbVehicles) + ' \t \t ' + str(capacity) + ' \n ')
        file.write('\n')
        file.write('CUSTOMER')
        file.write('\n')
        file.write('CUST NO. \t XCOORD.  \t YCOORD. \t DEMAND \t READY TIME \t DUE DATE \t SERVICE TIME \n')

        for node in nodes:
            line = " " + str(node['id']) + " \t \t " + str(node['x']) + " \t \t " + str(node['y']) + " \t \t " + str(node['demand']) + " \t \t " + str(node['ready']) + " \t \t " + str(node['due']) +  " \t \t " + str(node['service']) + " \n "
            file.write(line)

def generate_random_instance(size, total, seed):
    rd.seed(seed)
    #PossiblePositionDepot = ["Center", "Eccentric", "Random"]
    PossiblePositionDepot = ['Center', 'Random'] # depot is never eccentric in solomon instances
    PossibleHorizon = ["Large", "Small"]
    PossiblePositionCustomer = ["Random", "Cluster", "Mixt"]
    #PossiblePositionCustomer = ['Random'] # type of generation
    PossibleDemandDistribution = ["Large", "Medium", "Small"]
    PossibleWidthTW = ["Wide", "Tight"]
    PossibleServiceDistribution = ["Large", "Small"]

    # only the distribution is randomized in instances (and the position of the depot)
    for positionCustomer in PossiblePositionCustomer:
        for horizon in PossibleHorizon:
            for widthTW in PossibleWidthTW:
                for number in range(1, total+1):
                    if number < 10:
                        id = "0" + str(number)
                    else:
                        id = str(number)
                    positionDepot = rd.sample(PossiblePositionDepot, 1)[0]
                    #horizon = rd.sample(PossibleHorizon, 1)[0]
                    #positionCustomer = rd.sample(PossiblePositionCustomer, 1)[0]
                    demandDistribution = rd.sample(PossibleDemandDistribution, 1)[0]
                    #widthTW = rd.sample(PossibleWidthTW, 1)[0]
                    serviceDistribution = rd.sample(PossibleServiceDistribution, 1)[0]

                    #name_instance = positionDepot[0]+horizon[0]+positionCustomer[0]+demandDistribution[0]+widthTW[0]+serviceDistribution[0]+'_1'
                    short_name = positionCustomer[0] + horizon[0] + widthTW[0] + id
                    output_path = os.path.join("generated_instances", short_name)
                    entete = "Position Depot: " + positionDepot + "\n"
                    entete += "Horizon for Depot: " + horizon + "\n"
                    entete += "Position of customers: " + positionCustomer + "\n"
                    entete += "Distribution of the demand: " + demandDistribution + "\n"
                    entete += "Width of time windows: " + widthTW + "\n"
                    entete += "Distribution of the service: " + serviceDistribution + "\n"

                    v, c, n = generate_instance(size, positionDepot, horizon, positionCustomer, demandDistribution, widthTW, serviceDistribution)
                    write_instance(short_name, entete, output_path, v, c, n)

generate_random_instance(100, 8, 1)
