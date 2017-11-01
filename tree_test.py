import random
import operator
import csv
import itertools

import numpy
import eval_model as em #

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# defined a new primitive set for strongly typed GP
num_classifiers = 2
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float,num_classifiers), float, "IN")
#
#pset.addPrimitive(operator.add, [float,float], float)
#pset.addPrimitive(operator.sub, [float,float], float)
#pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(max, [float,float], float)
pset.addPrimitive(min, [float,float], float)
#pset.addPrimitive(operator.neg, [float],float)
pset.addPrimitive(operator.lt, [float, float], bool)
#pset.addPrimitive(operator.eq, [float, float], bool)

# terminals
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
#pset.addTerminal(0.0, float)
#pset.addTerminal(0.1, float)
#pset.addTerminal(0.2, float)
#pset.addTerminal(0.3, float)
#pset.addTerminal(0.4, float)
#pset.addTerminal(0.5, float)
#pset.addTerminal(0.6, float)
#pset.addTerminal(0.7, float)
#pset.addTerminal(0.8, float)
#pset.addTerminal(0.9, float)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# define eval(individual) as ensemble testing accuracy on a given dataset
def eval_mod(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    data_dir = '../converted/segmented/cropped/test/TREMULOUS/'
    model_dir = '/checkpoint/conv_vowel/TREMULOUS/'
    model_fn = ['fc_16_1000_model.ckpt','fc_64_1000_model.ckpt']
    x = [em.eval(model_dir+model_fn[i],data_dir) for i in range(num_classifiers)]
    #print(x)
    return func(x[0],x[1]),

toolbox.register("evaluate",eval_mod)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    #random.seed(10)
    pop = toolbox.population(n=10)  # 100
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 2, stats, halloffame=hof)  # 40
    for t in hof:
        print(str(t))

    return pop, stats, hof

if __name__ == "__main__":
    main()

