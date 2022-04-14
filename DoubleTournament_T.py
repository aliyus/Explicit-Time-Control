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

system = 'desktop'
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
from apgpdeap.apgp import apgpwGenStats, lastgenstats, gpSteadyState, apgpNoGenStats, gpDoubleT
from apgpdeap import tctools

run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #           (2)

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
#toolbox.register("select", tools.selTournament, tournsize=3) # ----------------------------?? breeding
#toolbox.register("select", tools.selDoubleTournament) # ----------------------------?? breeding
toolbox.register("select", tctools.selDoubleTournTime) # ----------------------------?? breeding
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# When an over the limit child is generated, it is simply replaced by a randomly selected parent.
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.register("worstfitness", tools.selWorst)
#===========================================================        

def main():
    random.seed(2019)

# ================================================
# ================================================
# GP Standard - Steady State
# ================================================
# ================================================
    tag = f'_Time_DoubleT_P5_EnergyHeat'#Standard GP - Steady State

# -----------------------------------------------------------------------
    if system == 'server':
        reportfolder = "/home/aliyu/Documents/Target/"
    elif system == 'laptop':
        reportfolder = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\TimeControl\\"
    elif system == 'desktop':
        reportfolder = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\TimeControl\\"

    report_csv = f"{reportfolder}{run_time}_{tag}.csv"
#-----------------------------------------------------------------------
#    
    for i in range(1, runs+1):
    # for i in range(runs+1):
        run = i
#        pop = toolbox.population(n=10)
        pop = toolbox.population(n=500)
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
        pop, log = gpDoubleT(pop, toolbox, 0.9, 0.1, 70, #              (9)
#        pop, log = gpDoubleT(pop, toolbox, 0.9, 0.1, 3, # 
                                 stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = 0.15)#0.017
#                                stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, tp = tp)
        print('Taking stats for the last generation....')
        #Collect stats for the last generation of each run.
        lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)

        
## ================================================
## ================================================
##       APGP No Generational Stats
## ================================================
## ================================================
#    poolsizelist = [5, 25, 50, 75, 100]
#    for poolsize in poolsizelist:
#        tag = f'nl_APGPnoGenStats_ps{poolsize}_P5_EnergyHeat'# APGP with NO GEN Stats - Steady State
##        report_csv = "/home/aliyu/Documents/Target/{run_time}_{tag}.csv"
##        report_csv = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Phase3\\Target\\{run_time}_{tag}.csv"
#        report_csv = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Phase3\\Target\\{run_time}_{tag}.csv"
#        for i in range(1, runs+1):          # < ---------------- ?????????????????????????????
#            run = i
##            pop = toolbox.population(n=50)  # < ---------------- ?????????????????????????????
#            pop = toolbox.population(n=500)
#            hof = tools.HallOfFame(1) 
#            #-----------------------------------------       
#            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
#            stats_size = tools.Statistics(len)
#            stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
#            stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
#            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
#            mstats.register("avg", numpy.mean)
#            mstats.register("std", numpy.std)
#            mstats.register("min", numpy.min)
#            mstats.register("max", numpy.max)
#            pop, log = apgpNoGenStats(pop, toolbox, 0.9, 0.1, 70, #              (9)       
##            pop, log = apgpNoGenStats(pop, toolbox, 0.9, 0.1, 3, #              (9)
#                                      stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, poolsize=poolsize, datatrain=datatrain, datatest=datatest, target = 0.025)
##                                    stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, tp = tp)
#            print('Taking stats for the last generation....')
#            #Collect stats for the last generation of each run.
#            lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN.......??  (9b)
#
#
## ================================================
## ================================================
## APGP With Generational Stats
## ================================================
## ================================================
#    poolsizelist = [5, 25, 50, 75, 100]
#    for poolsize in poolsizelist:
#        tag = f'nl_APGPwGenStat_ps{poolsize}_P5_EnergyHeat-GP'# APGP - with correlation checks
##        report_csv = "/home/aliyu/Documents/Target/{run_time}_{tag}.csv"
##        report_csv = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Phase3\\TargetBC\\{run_time}_{tag}.csv"
#        report_csv = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Phase3\\TargetBC\\{run_time}_{tag}.csv"
#        for i in range(1, runs+1):          # < ---------------- ?????????????????????????????
#            run = i
##            pop = toolbox.population(n=50)  # < ---------------- ?????????????????????????????
#            pop = toolbox.population(n=500)
#            hof = tools.HallOfFame(1) 
#            #-----------------------------------------       
#            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
#            stats_size = tools.Statistics(len)
#            stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
#            stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
#            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
#            mstats.register("avg", numpy.mean)
#            mstats.register("std", numpy.std)
#            mstats.register("min", numpy.min)
#            mstats.register("max", numpy.max)
#            pop, log = apgpwGenStats(pop, toolbox, 0.9, 0.1, 70, #              (9)
##            pop, log = apgpwGenStats(pop, toolbox, 0.9, 0.1, 3,      
#                                      stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, poolsize=poolsize, datatrain=datatrain, datatest=datatest, target = 0.025)
##                                    stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, tp = tp)
#            print('Taking stats for the last generation....')
#            #Collect stats for the last generation of each run.
#            lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN.......??  (9b)
#		
		
#==============================================================================
if __name__ == "__main__":
    main()    
#==============================================================================

