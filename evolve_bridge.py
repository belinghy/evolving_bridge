import random
import numpy as np
import matplotlib
from matplotlib.pyplot import plot, axis, arrow, xkcd, savefig, grid, clf
from deap import base, creator, tools, algorithms
from operator import attrgetter

import pdb
pdb.set_trace()

FOS_MIN = 1.5

# Predefined Joints
Pre = {}
Pre["Joints"] = np.array([[0,0,0], [2,0,0], [4,0,0], [6,0,0], [8,0,0], [10,0,0]])
# Predefined Load
Pre_force = -20000.0
Pre["Load"] = np.array([[0,0,0], [0,Pre_force,0], [0,Pre_force,0], [0,Pre_force,0], [0,Pre_force,0], [0,0,0]]).T

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def init_truss():
    truss = {}
    
    # Predefined joints
    truss["Joints"] = Pre["Joints"]

    return truss

toolbox = base.Toolbox()
toolbox.register("base_truss", init_truss)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.base_truss, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutAddNewJoint(individual):
    x = np.random.randint(0, 9)
    y = np.random.randint(1, 9)
    individual.append(x)
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
    Uff = np.linalg.solve(SSff, Loadff)
    
    ff = np.where(U.T==1)
    for i in range(len(ff[0])):
        U[ff[1][i], ff[0][i]] = Uff[i]
    
    F = np.sum(np.multiply(Tj, U[:, D["Beams"][1,:]] - U[:, D["Beams"][0,:]]), axis=0)
    R = np.sum(SS*U.T.flat[:], axis=1).reshape([w[1], w[0]]).T
    
    return F, U, R
    
    
def fos_eval(truss):
    D = {}
    
    M = len(truss["Beams"].T)
    N = len(truss["Joints"].T)
    
    # "1" means it is fixed, "0" means it is able to move, (x,y,z)
    D["Re"] = np.array([[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [1,1,1]]).T
    for _ in range(N-len(Pre["Joints"])):
        D["Re"] = np.column_stack([D["Re"], [0,0,1]])
        
    # Add loads, applying load to second and fourth beam in downward direction
    D["Load"] = Pre["Load"]
    for _ in range(N-len(Pre["Joints"])):
        D["Load"] = np.column_stack([D["Load"], [0,0,0]])

    # Add area information from truss
    D["A"] = []
    for member_size in truss["Sizes"]:
        D["A"].append(AREA_SEC[int(member_size)])
    D["Joints"] = truss["Joints"]
    D["Beams"] = truss["Beams"]
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
            FOS[i] = min(np.pi * np.pi * E * I_SEC[int(truss["Sizes"][i] - 1)] / (L[i]*L[i]) / -F[i], -FOS[i])
    
    count = 0
    for fos in FOS:
        if fos > FOS_MIN:
            count += 1
    
    return count,
    
toolbox.register("evaluate", fos_eval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutAddNewJoint)
toolbox.register("select", tools.selTournament, tournsize=3)

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.
    
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
    
    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.
    A first loop over :math:`P_\mathrm{o}` is executed to mate consecutive
    individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting
    :math:`P_\mathrm{o}` is returned.
    
    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    
    return offspring

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, history=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is 
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        
        # Keep track of best individual
        if history is not None:
            history.append(max(population, key=attrgetter("fitness")))
            
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream        

    return population, logbook

def main():
    import numpy
    
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    history = []
    pop, logbook = eaSimple(pop, toolbox, cxpb=0, mutpb=1, ngen=10, stats=stats, halloffame=hof, history=history, verbose=True)
    
    print history
    return pop, logbook, hof
 
if __name__ == "__main__":
    pop, log, hof = main()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    
    import matplotlib.pyplot as plt
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    raw_input()