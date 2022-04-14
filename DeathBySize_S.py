# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 00:24:30 2018
nls-
@author: ID915897
random seed toensure
"""
# Set number of runs

runs = 50
#runs = 2

#system = 'desktop'
#system = 'laptop'
#system = 'server'


# Initialise
import csv
import itertools
import operator
import math
import random
import numpy
#import deap
#from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import datetime
import time
from math import exp, cos, sin, log
#===============================================================================
#from . import tools
import pandas as pd
import numpy as np
import os
from functools import reduce
from operator import add, itemgetter
from numpy import arcsinh
from multiprocessing.pool import ThreadPool, threading

# apgptargetb.apgptarget 
#from apgptargetb.apgptarget import apgpwGenStats, lastgenstats, gpSteadyState, apgpNoGenStats, gpDoubleT
from apgpdeap.apgp import apgpwGenStats, lastgenstats, gpSteadyStateT, apgpNoGenStats, gpDoubleT, gpTDbS, gpSDbS
from apgpdeap import tctools

run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #           (2)
devicename = os.environ['COMPUTERNAME']
if devicename == 'DESKTOP-MLNSBQ2':
    system = 'laptop'
else: system = 'desktop'
#system = 'server'
"""    
#==============================================================================

from gptimecontrol import tools

import gptimecontrol 


#==============================================================================
"""
def div(left, right):
    return left / right
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #EnergyHeating

pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(div, [float,float], float) # ???????????????????
#pset.addPrimitive(math.log, [float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin,[float], float)

pset.addEphemeralConstant("nrand101", lambda: random.randint(1,100)/20, float)  #(3a)
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Weight is positive (i.e. maximising problem)  for normalised.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""

if system == 'server':
    datafolder = "/home/aliyu/Documents/CORR/Energy_Heating/energy_heating.csv"
elif system == 'laptop':
    datafolder = "C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Phase3\\Energy_Heating\\energy_heating.csv"
elif system == 'desktop':
    datafolder = "C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Phase3\\Energy_Heating\\energy_heating.csv"

with open(datafolder) as train:
    trainReader = csv.reader(train)
    Tpoints = list(list(float(item) for item in row) for row in trainReader)

#split data: random selection without replacement
#Tpoints = points.copy()
random.seed(2019)   
x1=random.shuffle(Tpoints)   
split = int(len(Tpoints)*0.2)
datatrain=Tpoints[split:len(Tpoints)]
datatest=Tpoints[0:split]


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""
#    Evaluate the mean squared error between the expression
def evalSymbReg(individual, datatrain, datatest):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    time_st = time.perf_counter()

    # Evaluate the mean squared error between the expression and the real function
    #Training Error - Fitness ===============================
    for z in range(3): #                                                       (5)
        error=0.
        total=0.
        try:
            for item in datatrain:
                total = total + ((func(*item[:8])) - (item[8]))**2                  #Energy Efficiency - Heat
                MSE = total/len(datatrain)
                error = 1/(1+ MSE)               
        except (ZeroDivisionError, ValueError):#    except ZeroDivisionError:
                error = 0.010101010101010101

    #Test Data =============================================
    error_test=0.
    total_t=0.
    try:
        for item in datatest:
            total_t = total_t + ((func(*item[:8])) - (item[8]))**2                  #Energy Efficiency - Heat
            MSE_t = total_t/len(datatest)
            error_test = 1/(1+ MSE_t)               
    except (ZeroDivisionError, ValueError):#    except ZeroDivisionError:
            error_test = 0.010101010101010101
            
    evln_sec=float((time.perf_counter() - time_st))
    return error, evln_sec, error_test

"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""
#============================================================
#============================================================
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3) # ----------------------------?? breeding
#toolbox.register("select", tools.selDoubleTournament) # ----------------------------?? breeding
#toolbox.register("select", tctools.selDoubleTournTime) # ----------------------------?? breeding
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# When an over the limit child is generated, it is simply replaced by a randomly selected parent.
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.register("worstfitness", tools.selWorst)
#===========================================================        

"""
============================================================================
Function to create initial population: (1) FIXED SIZE AND (2)  UNIQUE INDIVIDUALS
(Constants are treated as same).
============================================================================
"""
def inipopln():    
    ini_len = 10 # Initial lengths
    popsize = 500
    print(f'Creating a population of {popsize} individuals - each of size: {ini_len}')
    # Function to extract the node types   ----------------------------------------
    def graph(expr):
        str(expr)
        nodes = range(len(expr))
        edges = list()
        labels = dict()
        stack = []
        for i, node in enumerate(expr):
            if stack:
                edges.append((stack[-1][0], i))
                stack[-1][1] -= 1
            labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
            stack.append([i, node.arity])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return nodes, edges, labels
#    ------------------------------- create 1st individual
    current=[]
    newind=[]
    newind= toolbox.population(n=1)
    while len(newind[0]) != ini_len:
        newind = toolbox.population(n=1)
    current.append(newind[0])
#    ------------------------------- Create others; 
#    For each new one check to see a similar individual exists in the population.
    while len(current) < popsize:
        pop = toolbox.population(n=1)
        if len(pop[0]) == ini_len:
            # ----------------------------- Check for duplicate
            lnodes, ledges, llabels = graph(pop[0])
            similarity = 'same'
            for k in range(len(current)): # CHECK all INDs in CURRENT population
                nodes, edges, labels = graph(current[k])
                for j in range(len(labels)): # Check NEW against IND from CURRENT
                    constants = 'no' # will use to flag constants
                    if labels[j] != llabels[j]: 
                        similarity = 'different' 
                        # no need to check other nodes as soon as difference is detected 
                    if '.' in str(labels[j]) and '.' in str(llabels[j]): constants = 'yes'
                    if labels[j] != llabels[j] or constants != 'yes': # They are different and not constants
                        continue # no need to check other nodes as soon as difference is detected 
                if similarity =='same': # skips other checks as soon as it finds a match
                    continue
            if similarity == 'different': # add only if different from all existing
                current.append(pop[0])     
    print('population created')
    return current
"""
============================================================================
============================================================================
"""
def main():
    random.seed(2019)

# ================================================
# ================================================
# GP + DeathbySize - Steady State
# ================================================
# ================================================
    tag = f'_FL_Size_DeathbySize_P5_EnergyHeat'#Standard GP - Steady State

# -----------------------------------------------------------------------
    if system == 'server':
        reportfolder = "/home/aliyu/Documents/Target/"
    elif system == 'laptop':
        reportfolder = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_TC_DeathBySize\\"
    elif system == 'desktop':
        reportfolder = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_TC_DeathBySize\\"

    report_csv = f"{reportfolder}{run_time}_{tag}.csv"
#-----------------------------------------------------------------------
#    
    for i in range(1, runs+1):
    # for i in range(runs+1):
        run = i
#        pop = toolbox.population(n=10)
#        pop = toolbox.population(n=500)
#        pop = inipopln()
        pop = inipopln()
        hof = tools.HallOfFame(1) 
        #-----------------------------------------       
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
        stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        pop, log = gpSDbS(pop, toolbox, 0.9, 0.1, 70, #              (9)
#        pop, log = gpDoubleT(pop, toolbox, 0.9, 0.1, 3, # 
                                 stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = 0.16)#0.017
#                                stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, tp = tp)
        print('Taking stats for the last generation....')
        #Collect stats for the last generation of each run.
        lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)

        

# ================================================
# ================================================
# GP Standard - Steady State
# ================================================
## ================================================
    tag = f'FL_standardGP_P5_EnergyHeat'#Standard GP - Steady State
# -----------------------------------------------------------------------
    if system == 'server':
        reportfolder = "/home/aliyu/Documents/Target/"
    elif system == 'laptop':
        reportfolder = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_TC_DeathBySize\\"
    elif system == 'desktop':
        reportfolder = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_TC_DeathBySize\\"

    report_csv = f"{reportfolder}{run_time}_{tag}.csv"
#-----------------------------------------------------------------------
    for i in range(1, runs+1):
    # for i in range(runs+1):
        run = i
#        pop = toolbox.population(n=10)
#        pop = toolbox.population(n=500)
#        pop = inipopln()
        pop = inipopln()
        hof = tools.HallOfFame(1) 
        #-----------------------------------------       
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
        stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        pop, log = gpSteadyStateT(pop, toolbox, 0.9, 0.1, 70, #              (9)
#        pop, log = gpSteadyStateT(pop, toolbox, 0.9, 0.1, 3, # 
                                 stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = 0.16)
#                                stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, tp = tp)
        print('Taking stats for the last generation....')
        #Collect stats for the last generation of each run.
        lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)
			
#==============================================================================
if __name__ == "__main__":
    main()    
#==============================================================================

