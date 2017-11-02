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
num_classes = 4

#
class In_type(tuple):
    def __init__(self):
        self.list = [(0.0,)*num_classes,(0.0,)*num_classes]
    def __iter__(self):
        return iter(self.list)

#
class Out_type(object):
    def __init__(self):
        self.list = (0.0,)*num_classes
    def __iter__(self):
        return iter(self.list)

in_type = [(float,)*num_classes,(float,)*num_classes]
out_type = (float,)*num_classes
mul_in_type = [(float,)*num_classes,float]
#
#
#pset = gp.PrimitiveSetTyped("MAIN",In_type,out_type,"IN")   # itertools.repeat
pset = gp.PrimitiveSet("MAIN",2)
#
def add_(a,b):
    if not (type(a) is list) or not (type(b) is list):
        return None
    if len(a) != len(b):
        return None
    s = []
    len_ = len(a)
    for i in range(len_):
        s.append(a[i]+b[i])
    return s

def mul_(a,c):
    if not (type(a) is list) or not (type(c) is float):
        return None
    s = []
    for e in a:
        s.append(e*c)
    return s

pset.addPrimitive(add_, 2)
#pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(mul_, 2)
#pset.addPrimitive(max, [float,float], float)
#pset.addPrimitive(min, [float,float], float)
#pset.addPrimitive(operator.neg, [float],float)
#pset.addPrimitive(operator.lt, [float, float], bool)
#pset.addPrimitive(operator.eq, [float, float], bool)

# terminals
#pset.addTerminal(False, bool)
#pset.addTerminal(True, bool)
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
    t_x,t_y = em.get_test(data_dir)
    models_output = [em.eval(t_x,model_dir+model_fn[i],data_dir) for i in range(num_classifiers)]
    ens_output = []
    for i in range(len(models_output[0])):
        ens_output.append(func(models_output[0][i],models_output[1][i]))
    # Compute ensemble accuracy
    
    return random.random(),

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

