import random
import operator
import csv
import itertools

import numpy
import eval_model as em #
import tensorflow as tf

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from scoop import futures

# defined a new primitive set for strongly typed GP
num_classifiers = 2
num_classes = 4
#
#
#pset = gp.PrimitiveSetTyped("MAIN",list,list,"IN")   # itertools.repeat
pset = gp.PrimitiveSet("MAIN",2)
#
def listify(a):
    if not (type(a) is list):   # a is constant
        return None #[a]*num_classes
    else:
        return a

#
def constify(c):
    if not(type(c) is float):   # c is list
        return None #c[0]
    else:
        return c

#
def add_(a,b):
    a = listify(a)
    b = listify(b)
    if a is None or b is None:
        return None
    s = []
    for i in range(len(a)):
        s.append(a[i]+b[i])
    return s

#
def mul_(a,c):
    a = listify(a)
    c = constify(c)
    if a is None or c is None:
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
toolbox.register("map",futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval_mod(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    data_dir = '../converted/segmented/cropped/test/TREMULOUS/'
    model_dir = '/checkpoint/conv_vowel/TREMULOUS/'
    model_fn = ['fc_16_1000_model.ckpt','fc_64_1000_model.ckpt']
    t_x,t_y = em.get_test(data_dir)
    with tf.Session() as sess:
        models_output = [em.eval(sess,t_x,model_dir+model_fn[i],data_dir).tolist() for i in range(num_classifiers)]
    ens_output = []
    for i in range(len(models_output[0])):
        res = func(models_output[0][i],models_output[1][i])
        if res is None:
            return 0.0,
        ens_output.append(res)
    # Compute ensemble accuracy
    acc = 0.0
    for i in range(len(ens_output)):
        c1 = numpy.argmax(ens_output[i])
        c2 = numpy.argmax(t_y[i])
        if c1 == c2:
            acc += 1.0
    return acc/len(ens_output),

toolbox.register("evaluate",eval_mod)
toolbox.register("select", tools.selTournament, tournsize=2)    # 3
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    #random.seed(10)
    pop = toolbox.population(n=12)  # 100
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.2, 0.1, 30, stats, halloffame=hof)  # 40
    for t in hof:
        print(str(t))

    return pop, stats, hof

if __name__ == "__main__":
    main()

