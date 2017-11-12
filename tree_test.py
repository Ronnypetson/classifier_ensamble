import random
import operator
import csv
import itertools
import os
import numpy
import eval_model as em #

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
#from scoop import futures

#
def listify(a):
    if type(a) is list:
        return a
    else:   # a is constant
        return None

#
def constify(c):
    if type(c) is float:
        return c
    else:   # c is list
        return None

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

def eval_mod(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    ens_output = []
    for i in range(len(models_output[0])):
        res = func(models_output[0][i],models_output[1][i],models_output[2][i],models_output[3][i])
        if res is None or type(res) is float:
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

# defined a new primitive set for strongly typed GP
num_classifiers = 4
num_classes = 4 ## 2
writer = 'TREMULOUS/'
authors_bin = ['TREMULOUS/','NON-TREMULOUS/','Thorpe/'] ##
img_dir_bin = '../converted/segmented/cropped/'
data_dir = '../converted/segmented/cropped/test/'+writer
model_dir = '/checkpoint/conv_vowel/'+writer
model_fn = ['fc_16_5000_model.ckpt','fc_64_5000_model.ckpt','cl_16_5000_model.ckpt','cl_64_5000_model.ckpt']
#
#
#pset = gp.PrimitiveSetTyped("MAIN",list,list,"IN")   # itertools.repeat
pset = gp.PrimitiveSet("MAIN",num_classifiers)    # 2

pset.addPrimitive(add_, 2)
pset.addPrimitive(mul_, 2)
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
#toolbox.register("map",futures.map)
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)  # max_=2, genHalfAndHalf
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate",eval_mod)
toolbox.register("select", tools.selTournament,tournsize=2)    # 3
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#
t_x, t_y = em.get_test(data_dir)    ## em.get_test_bin(authors_bin,img_dir_bin)
models_output = [em.eval(t_x,model_dir+model_fn[i],data_dir).tolist() for i in range(num_classifiers)]
pop = toolbox.population(n=30)  # 100
hof = tools.HallOfFame(10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 20, stats, halloffame=hof)  # 40
for t in hof:
    print(str(t),eval_mod(t))

#if __name__ == "__main__":
#    main()

