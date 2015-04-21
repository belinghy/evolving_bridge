from deap import base
from deap import creator
from deap import tools

import numpy as np
import matplotlib
from matplotlib.pyplot import plot, axis, arrow, xkcd, savefig, grid, clf
from operator import attrgetter

import pdb
pdb.set_trace()

# Yield strength of steel
Fy = 344 * pow(10,6)
# Elastic modulus of steel
E = 210 * pow(10, 9)
# Outer diameters of some optional sizes, in meters
OUTER_DIAM = [(x+1.0)/100 for x in range(10)]
# Cross sectional area in m^2
AREA_SEC = [np.pi*pow(d/2, 2) - np.pi*pow(d/2-d/15, 2) for d in OUTER_DIAM]
# Moment of inertia, in m^4
I_SEC = [np.pi*(pow(d, 4) - pow((d - 2*d/15), 4))/64 for d in OUTER_DIAM]
# Weight per length, kg/m
WEIGHT = [a*7870 for a in AREA_SEC]
# Factor of safety
FOS_MIN = 1.5

# Predefined Joints
Pre = {}
Pre["Joints"] = np.array([[0,0,0], [2,0,0], [4,0,0], [6,0,0], [8,0,0], [10,0,0], [1,2,0]])
# Predefined Load
Pre_force = -20000.0
Pre["Load"] = np.array([[0,0,0], [0,Pre_force,0], [0,Pre_force,0], [0,Pre_force,0], [0,Pre_force,0], [0,0,0], [0,0,0]])
# Predefined allowed movement
Pre["Re"] = np.array([[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [1,1,1], [0,0,1]])
Pre["Edges"] = np.array([[i,i+1] for i in range(len(Pre["Joints"])-1)])

# Window size
X_MAX = 10
Y_MAX = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def init_truss():
    truss = {}
    truss["Joints"] = Pre["Joints"].T
    return truss
    

toolbox = base.Toolbox()
toolbox.register("base_truss", init_truss)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.base_truss, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutAddNewJoint(individual):
    x = np.random.randint(0, X_MAX)
    y = np.random.randint(1, Y_MAX)
    individual[0]["Joints"] = np.column_stack([individual[0]["Joints"], [x,y,0]])
    return individual,

def force_eval(D):
    """
    This function takes as input a dictionary D that defines the following variables:
        "Re":    encodes the permitted motion of each joint. Each joint is represented by 
                 a vector of three binary variables, indicating whether or not it is supported
                 in the, x, y and z direction.
        "Joints": This indicates the x, y, and z coordinates of each joint.
        "Load":  This indicates the x, y and z loadings placed on each joint.
        "Beams":   This indicates how the joints are connected
        "E":     This indicates the elastic modullus of each connecting member described above.
        "A":     This indicates the area of each connecting member described above
        
    The function returns three variables.  
        F:       The forces present in each member
        U:       Node displacements
        R:       Node reactions
    """
    Tj = np.zeros([3, np.size(D["Beams"], axis=1)])
    w = np.array([np.size(D["Re"], axis=0), np.size(D["Re"], axis=1)])
    SS = np.zeros([3*w[1], 3*w[1]])
    U = 1.0 - D["Re"]
    
    # This identifies joints that are unsupported, and can therefore be loaded
    ff = np.where(U.T.flat == 1)[0]
    
    # Step through the each member in the truss, and build the global stiffness matrix
    for i in range(np.size(D["Beams"], axis=1)):
        H = D["Beams"][:, i]
        C = D["Joints"][:, H[1]] - D["Joints"][:, H[0]]
        Le = np.linalg.norm(C)
        T = C/Le
        s = np.outer(T, T)
        G = D["E"][i]*D["A"][i]/Le
        ss = G*np.concatenate((np.concatenate((s, -s), axis=1), np.concatenate((-s, s), axis=1)), axis=0)
        Tj[:, i] = G*T
        e = range((3*H[0]), (3*H[0] + 3)) + range((3*H[1]), (3*H[1] + 3))
        for ii in range(6):
            for j in range(6):
                SS[e[ii], e[j]] += ss[ii, j]

    SSff = np.zeros([len(ff), len(ff)])
    for i in range(len(ff)):
        for j in range(len(ff)):
            SSff[i,j] = SS[ff[i], ff[j]]
                
    Loadff = D["Load"].T.flat[ff]
    try:
        Uff = np.linalg.solve(SSff, Loadff)
    except LinAlgError:
        F = np.ones(D["Beams"].shape[1]) * float('inf')
        U = np.ones((D["Joints"].shape[1], D["Joints"].shape[1]))
        R = np.ones((D["Joints"].shape[1], D["Joints"].shape[1]))
        return F, U, R
        
    ff = np.where(U.T==1)
    for i in range(len(ff[0])):
        U[ff[1][i], ff[0][i]] = Uff[i]
    
    F = np.sum(np.multiply(Tj, U[:, D["Beams"][1,:]] - U[:, D["Beams"][0,:]]), axis=0)
    R = np.sum(SS*U.T.flat[:], axis=1).reshape([w[1], w[0]]).T
    
    return F, U, R
    
    
def fos_eval(truss):
    D = {}
    temp = {}
    temp["Joints"] = truss[0]["Joints"]
    
    # Initialize some stuff
    edges = matplotlib.tri.Triangulation(temp["Joints"][0, :], temp["Joints"][1, :]).edges
    temp["Beams"] = edges.T
    temp["Sizes"] = np.ones(len(temp["Beams"].T))*4 # Choose the 4th one in AREA_SEC
    
    M = len(temp["Beams"].T)
    N = len(temp["Joints"].T)
    
    # "1" means it is fixed, "0" means it is able to move, (x,y,z)
    D["Re"] = Pre["Re"].T
    for _ in range(N-len(Pre["Joints"])):
        D["Re"] = np.column_stack([D["Re"], [0,0,1]])
        
    # Add loads, applying load to second and fourth beam in downward direction
    D["Load"] = Pre["Load"].T
    for _ in range(N-len(Pre["Joints"])):
        D["Load"] = np.column_stack([D["Load"], [0,0,0]])

    # Add area information from truss
    D["A"] = []
    for member_size in temp["Sizes"]:
        D["A"].append(AREA_SEC[int(member_size)])
    D["Joints"] = temp["Joints"]
    D["Beams"] = temp["Beams"]
    D["E"] = E*np.ones(M)
    
    # Do force analysis
    F, U, R = force_eval(D)
    
    # Calculate lengths
    L = np.zeros(M)
    for i in range(M):
        L[i] = np.linalg.norm(D["Joints"][:, D["Beams"][0, i]] - D["Joints"][:, D["Beams"][1, i]])
    
    # Calculate FOS's
    FOS = np.zeros(M)
    for i in range(len(F)):
        FOS[i] = D["A"][i] * Fy / F[i]
        if FOS[i] < 0:
            FOS[i] = min(np.pi * np.pi * E * I_SEC[int(temp["Sizes"][i] - 1)] / (L[i]*L[i]) / -F[i], -FOS[i])
    
    count = 0
    for fos in FOS:
        if fos > FOS_MIN:
            count += 1
    
    return count,

# Operator registering
toolbox.register("evaluate", fos_eval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutAddNewJoint)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    
    pop = toolbox.population(n=10)
    CXPB, MUTPB, NGEN = 0.0, 1.0, 10
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()