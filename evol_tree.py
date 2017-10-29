import os, random
import operator
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

pset = gp.PrimitiveSet("main", 3)
#
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addTerminal(0.0)
pset.addTerminal(0.1)
pset.addTerminal(0.2)
pset.addTerminal(0.3)
pset.addTerminal(0.4)
pset.addTerminal(0.5)
pset.addTerminal(0.6)
pset.addTerminal(0.7)
pset.addTerminal(0.8)
pset.addTerminal(0.9)
#
pset.renameArguments(ARG0="c1")
pset.renameArguments(ARG1="c2")
pset.renameArguments(ARG2="c3")
#
expr_ = gp.genFull(pset, min_=1, max_=3)
#print(expr_)
#tree = gp.PrimitiveTree(expr_)
#print(str(tree))    # generates random tree
#function = gp.compile(tree,pset)
#print(function(1,2,0))

creator.create("FitnessMin",base.Fitness,weights=(1.0,))
creator.create("Individual",gp.PrimitiveTree,fitness=creator.FitnessMin,pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr_",gp.genHalfAndHalf,pset=pset,min_=1,max_=2)    # gp.genHalfAndHalf
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.expr_)
toolbox.register("population",tools.initRepeat,list,toolbox.individual) # gp.PrimitiveTree
toolbox.register("compile",gp.compile,pset=pset)

# define eval(individual) as ensemble testing accuracy on a given dataset
def evaluate_(individual):
    return random.random(),  #

#def mutate_(individual):
#    return gp.mutNodeReplacement(individual,expr_)    # ,expr_,pset

# http://deap.gel.ulaval.ca/doc/dev/api/tools.html
toolbox.register("evaluate",evaluate_)
toolbox.register("select",tools.selTournament,tournsize=3)    # tournsize=2,
toolbox.register("mate",tools.cxOnePoint)
toolbox.register("expr_mut",gp.genFull,min_=0,max_=2)
toolbox.register("mutate",gp.mutUniform,expr=toolbox.expr_mut,pset=pset)

def main():
    #random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    #print(pop)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg",numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop,toolbox, 0.5, 0.2, 40, stats, halloffame=hof)
    
    return pop, stats, hof
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #CXPB, MUTPB = 0.5, 0.2
    
    #print("Start of evolution")
    
    # Evaluate the entire population
    #fitnesses = list(map(toolbox.evaluate, pop))
    #for ind, fit in zip(pop, fitnesses):
    #    ind.fitness.values = fit
    
    #print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    #fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    #g = 0
    
    # Begin the evolution
    #while max(fits) < 100 and g < 1000:
        # A new generation
    #    g = g + 1
    #    print("-- Generation %i --" % g)
        
        # Select the next generation individuals
    #    offspring = toolbox.select(pop) # len(pop),1
        # Clone the selected individuals
    #    offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
    #    for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
    #        if random.random() < CXPB:
    #           toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
    #            del child1.fitness.values
    #            del child2.fitness.values

    #    for mutant in offspring:

            # mutate an individual with probability MUTPB
    #        if random.random() < MUTPB:
                #print(mutant)
    #            toolbox.mutate(mutant)
    #            del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
    #    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #    fitnesses = map(toolbox.evaluate, invalid_ind)
    #    for ind, fit in zip(invalid_ind, fitnesses):
    #        ind.fitness.values = fit
        
    #    print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
    #    pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
    #    fits = [ind.fitness.values[0] for ind in pop]
        
    #    length = len(pop)
    #    mean = sum(fits) / length
    #    sum2 = sum(x*x for x in fits)
    #    std = abs(sum2 / length - mean**2)**0.5
        
    #    print("  Min %s" % min(fits))
    #    print("  Max %s" % max(fits))
    #    print("  Avg %s" % mean)
    #    print("  Std %s" % std)
    
    #print("-- End of (successful) evolution --")
    
    #best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()

