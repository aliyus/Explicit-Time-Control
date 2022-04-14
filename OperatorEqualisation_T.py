# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 00:24:30 2018
nls-
@author: ID915897
random seed toensure
"""
settarget = 0.16
bin_width = 5

# Set number of runs
#runs = 2
runs = 50

#system = 'desktop'
#system = 'laptop'
#system = 'server'

import csv
import itertools
# Initialise
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
# from apgptargetb.apgptarget import apgpwGenStats, lastgenstats, gpSteadyState, apgpNoGenStats, gpDoubleT
from deap import tools
#from multiprocessing.pool import ThreadPool, threading
import random
from operator import add, itemgetter
import numpy as np
import pandas as pd
import os
from functools import reduce

run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #           (2)
devicename = os.environ['COMPUTERNAME']
if devicename == 'DESKTOP-MLNSBQ2':
    system = 'laptop'
else: system = 'desktop'



"""
OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
"""    
#==============================================================================

from math import ceil

def gpOpEqT(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None,
             verbose=__debug__, run=1, datatrain=None, datatest=None, report_csv=None, method = None, bin_width=None, t_bin_width=None, target =None):      # <--- opeq (width)
    """
    This algorithm implements Operator Equalisation and uses a generational GP.
    Parameters::
    method == 'DynOpEq' or 'MutOpEq' 
    bin_width= 
    """  
    poplnsize = len(population)
    counteval =0
    popsize = len(population)
    mettarget = 0 # 0 = not set

    
    def updatehof():    
        nonlocal population, bin_tracker, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv#, pset 
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

#        print(halloffame[0].fitness.values[0])

#    update_lock = threading.Lock()
#    counteval_lock = threading.Lock()
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
#+++++++++++++++++++++++++++++++++++++++++++++
#Evaluation of Initial Population
#+++++++++++++++++++++++++++++++++++++++++++++
    # Evaluate the individuals with an invalid fitness       
    for ind in population:
        if not ind.fitness.valid:
            xo, yo, zo = toolbox.evaluate(ind, datatrain, datatest)
            ind.evlntime = yo,
            ind.testfitness = zo,
            ind.fitness.values = xo,
            if ind.fitness.values == (0.0101010101010101,) :
                ind.fitness.values = 0.0, #for maximising
            if ind.testfitness == (0.0101010101010101,) :
                ind.testfitness = 0.0, #for maximising                
#                print('check this out')
#                print(str(ind))
#                print(str(ind.fitness.values))
    
    #++++++ Update Hof +++++++++++++++++
    updatehof()
#    try:
#        halloffame.update(population)
#    except AttributeError:
#        pass

#   halloffame[0].fitness.values[0]
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
#    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0])])
    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                   halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
    #+++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++
    
    #==================================== BIN INITIAL POPULATION =====================
    # Based on evlntime only
    
    #population = toolbox.population(popsize)
    bin_tracker = pd.DataFrame(columns=['ID', 'Capacity', 'Current', 'Max_fitness', 'Avg_Fitness'])

# ---------------------------
    unit = t_bin_width/bin_width #[[[[[[[[[[[[[ t_bin_width parameter of OpEQT ]]]]]]]]]]]]]
# ---------------------------


    for ind in population:  
#        b = int((len(ind) - 1)/bin_width + 1)
        b = int( ( ceil(ind.evlntime[0]/unit) - 1)/bin_width + 1 ) #[[[[[[[[[[[[[[[[[[ change ]]]]]]]]]]]]]]]]]]       

        if (bin_tracker['ID']== b).any():
          
            bin_tracker.loc[b,'Current']= bin_tracker['Current'][b] +1
        else:
            bin_tracker.loc[b] = [b, 1, 1, 0.0, 0.0]

            upboundary = bin_tracker['ID'].max()
            loboundary = bin_tracker['ID'].min()     
          
            # If new is outside boundarys: 
            # Then make available other bins between new bin and boundaries (wih capacity 1)
            if b > upboundary:
#                    print(f'create inbetween bins - from {upboundary} to {b}')
                nb = b - 1
                while nb > upboundary:
                    bin_tracker.loc[nb] = [nb, 1, 1, 0.0, 0.0]
                    nb -= 1
                    print(f'added upper bin: {nb} - bins - from {upboundary} to {b}')
            elif b < loboundary:
#                    print(f'create inbetween bins - from {loboundary} to {b}')
                nb = b + 1
                while nb < loboundary:
                    bin_tracker.loc[nb] = [nb, 1, 1, 0.0, 0.0]
                    nb += 1
                    print(f'added lower bin: {nb}- from {loboundary} to {b}')

#    =================================================================================
#    =================================================================================
            # TIME EQUIVALENT
#    =================================================================================
#    =================================================================================

#==================== UPDATE BIN with MAX & AVERAGE =================================
    def updatebintracker():
        nonlocal population, bin_tracker, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 
        
#        for j in range(len(bin_tracker['ID'])):# print(bin_tracker.iloc[j]['ID'])
#            b = bin_tracker.iloc[j]['ID']
        for j in bin_tracker['ID']:# print(bin_tracker.loc[j]['ID'])
            b = bin_tracker.loc[j]['ID']
            max_bin_fitness = 0
            avg_bin_fitness = 0
            bincount = 0
            totalbinfitness = 0
            # Iterate through population and calculate average, and max for each bin
            for i in range(len(population)): 
                # If evaluation time of i falls in range of bin?        
                if population[i].evlntime[0]  > (b-1)*t_bin_width and population[i].evlntime[0] <= (b*t_bin_width):
                    totalbinfitness = totalbinfitness + population[i].fitness.values[0]
#                    if totalbinfitness != 0 : print(totalbinfitness)
                    bincount += 1
                    if population[i].fitness.values[0] > max_bin_fitness:
                        max_bin_fitness = population[i].fitness.values[0]
                        print(f'New Bin {b} Max Fitness: {max_bin_fitness}')
            # Calculate the average
            # handle exception when bin is empty (does not apply for initial population)
            try:
                avg_bin_fitness = totalbinfitness/bincount
            except (IndexError, ZeroDivisionError):
                print('zero or index error')
                avg_bin_fitness = 0
                
            # Update the bins average and max values
            bin_tracker.loc[b,"Avg_Fitness"]= avg_bin_fitness
            bin_tracker.loc[b,"Max_fitness"]= max_bin_fitness
#==============================================================================

##==================== UPDATE BIN with MAX & AVERAGE =================================
#    def updatebintracker():
#        nonlocal population, bin_tracker, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 
#        
##        for j in range(len(bin_tracker['ID'])):# print(bin_tracker.iloc[j]['ID'])
##            b = bin_tracker.iloc[j]['ID']
#        for j in bin_tracker['ID']:# print(bin_tracker.loc[j]['ID'])
#            b = bin_tracker.loc[j]['ID']
#            max_bin_fitness = 0
#            avg_bin_fitness = 0
#            bincount = 0
#            totalbinfitness = 0
#            # Iterate through population and calculate average, and max for each bin
#            for i in range(len(population)): 
#                # If i falls in range of bin?
#                if len(population[i]) >= (b*bin_width - (bin_width-1)) and len(population[i]) <= b*bin_width:
#        #            print(len(population[i]))
#                    totalbinfitness = totalbinfitness + population[i].fitness.values[0]
#                    bincount += 1
#                    if population[i].fitness.values[0] > max_bin_fitness:
#                        max_bin_fitness = population[i].fitness.values[0]
#                        print(f'New Bin {b} Max Fitness: {max_bin_fitness}')
#            # Calculate the average
#            # handle exception when bin is empty (does not apply for initial population)
#            try:
#                avg_bin_fitness = totalbinfitness/bincount
#            except (IndexError, ZeroDivisionError):
#                print('zero or index error')
#                avg_bin_fitness = 0
##            print(f'Bin {b} average fitness = {avg_bin_fitness}')        
#
##            print(f'Average for bin {b} : {avg_bin_fitness}')
##            bin_tracker.set_value(b, 'Avg_Fitness', avg_bin_fitness)
##            bin_tracker.loc[b,"Avg_Fitness"]= avg_bin_fitness
#            bin_tracker.loc[b,"Avg_Fitness"]= avg_bin_fitness
#
#                           
##            print(f'Current Max: {max_bin_fitness}')
##            bin_tracker.set_value(b, 'Max_fitness', max_bin_fitness)
#            bin_tracker.loc[b,"Max_fitness"]= max_bin_fitness
##            bin_tracker.loc[b]["Max_fitness"]= max_bin_fitness
##==============================================================================

# ==================== UPDATE BIN CAPACITY BASED ON FITNESS AVGs ==============
    def updatebincapacity():
        nonlocal population, bin_tracker, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 
        
        for j in bin_tracker['ID']: #print(j)
#            b = int(bin_tracker.iloc[j]['ID'])
#            b = int(bin_tracker.loc[j,'ID'])
            b = bin_tracker.loc[j,'ID']
            new_capacity= round(len(population)*(bin_tracker.loc[j]['Avg_Fitness']/(bin_tracker['Avg_Fitness'].sum())))
#            bin_tracker.set_value(b, 'Capacity', new_capacity)
            bin_tracker.loc[b, "Capacity"] = new_capacity
        while bin_tracker['Capacity'].sum() < len(population):
#            print('???')
#            print('DISPARITY between BIN CAPACITY and POPULATION SIZE!')
#            print(bin_tracker)
            bin_tracker.loc[b, "Capacity"] += 1 # rounded calculation of capacity may lead to shortage by 1 - > increment capacity to round up to population size.
        print(bin_tracker)

# ===================== UPDATE BIN TRACKER FOR INITIAL POPULATION ============= 
    updatebintracker()
    updatebincapacity()



    # PRODUCE TWO OFFSPRING & UPDATE THEIR FITNESS VALUES
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, target, mettarget #, update_lock, counteval_lock, poolsize
        
        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
#        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))
#        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=2, fitness_first=True, fit_attr='fitness')))

        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values

        #++++++++ mutation on the offspring ++++++++++++++++               
        if random.random() < mutpb:
            p1, = toolbox.mutate(p1)
            del p1.fitness.values

        # Evaluate the offspring if it has changed
        if not p1.fitness.valid:
            #++++++++ Counting evaluations +++++++++++++++++
#            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
#            counteval_lock.release()
            xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
#            xo, yo, zo = toolbox.evaluate(p1)
            p1.evlntime = yo,
            p1.testfitness = zo,
            p1.fitness.values = xo, 
            #Check if ZeroDivisionError, ValueError 
            if p1.fitness.values == (0.0101010101010101,) :
                p1.fitness.values = 0.0, #for maximising
            if p1.testfitness == (0.0101010101010101,) :
                p1.testfitness = 0.0, #for maximising  
#        return p1

    			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
    			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    			#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
            if float(p1.testfitness[0]) >= target:
#                print('Hi')
                if mettarget == 0:
                    mettarget = counteval
                    print(f'Target met: {counteval}')
                    print(f'Test Fitness: {float(p1.testfitness[0])}')
                    targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.testfitness[0]), 'Met_at': mettarget}, index = {run})
                
                    target_csv = f'{report_csv[:-4]}_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(target_csv):
                        targetmet_df.to_csv(target_csv, mode='a', header=False)
                    else:
                        targetmet_df.to_csv(target_csv)                    
			#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]   
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
    
#        return p1, p2
        return p1
        
    
# Validation Funcion
    def f_validate(ind):
#        try:
        nonlocal bin_tracker, method, counteval
        # -----------------------------------------------------------
        # Identify the bin the ind should belong to based on length.
        # -----------------------------------------------------------
#            b = int((len(ind) - 1)/bin_width + 1) # ============================== TIME EQUIV.
        b = int( ( ceil(ind.evlntime[0]/unit) - 1)/bin_width + 1 ) #[[[[[[[[[[[[[[[[[[ change ]]]]]]]]]]]]]]]]]]
        # -----------------------------------------------------------
        # If there is room in the bin admit. 
        # For DynOpEq: If bin is full but ind is best of run admit
        # For MutOpEq: If bin full treated differently - see below
        # -----------------------------------------------------------
        if (bin_tracker['ID']== b).any():
            # check if full
            if bin_tracker['Current'][b] >= bin_tracker['Capacity'][b]:
#                if bin_tracker['Current'][b] <= bin_tracker['Capacity'][b]: #-??????????
                validate = False
                if method == 'DynOpEq' and ind.fitness.values[0] > bin_tracker['Max_fitness'][b]:
                    validate = True
            else:
                validate = True
        # -----------------------------------------------------------
        # If outside the current bins and best of run admit --> create bins between the new and lower or upper boundary if necessary
        # -----------------------------------------------------------    
        elif ind.fitness.values[0] > halloffame[0].fitness.values[0]:
            upboundary = bin_tracker['ID'].max()
            loboundary = bin_tracker['ID'].min()     
            
            bin_tracker.loc[b] = [b, 1, 1, ind.fitness.values[0], ind.fitness.values[0]]
            validate = True
            
            # If new is outside boundarys: 
            # Then make available other bins between new bin and boundaries (wih capacity 1)
            if b > upboundary:
#                    print(f'create inbetween bins - from {upboundary} to {b}')
                nb = b - 1
                while nb > upboundary:
                    bin_tracker.loc[nb] = [nb, 1, 1, 0.0, 0.0]
                    nb -= 1
                    print(f'added upper bin: {nb} - bins - from {upboundary} to {b}')
            elif b < loboundary:
#                    print(f'create inbetween bins - from {loboundary} to {b}')
                nb = b + 1
                while nb < loboundary:
                    bin_tracker.loc[nb] = [nb, 1, 1, 0.0, 0.0]
                    nb += 1
                    print(f'added lower bin: {nb}- from {loboundary} to {b}')
        else:
            validate = False
            if method == 'DynOpEq':
                print('new ind. dropped - no free bin & not better than hof')

        # -----------------------------------------------------------
        # For MutOpEq: If bin full mutate individual till it fits a bin
        # Evaluate: If new individual faulty (e.g zero division) drop and move on
        # -----------------------------------------------------------
       
        if method == 'MutOpEq' and validate == False:
#            if bin_tracker['Current'][b] >= bin_tracker['Capacity'][b]: # bin full?
        # -----------------------------------------------------------
        # Find the closest available neighbour (smaller prefered if same distance)
        # -----------------------------------------------------------                
#            print('1 - Check for nearest non-full bin:')
            count = bin_tracker['ID'].count() # no. of neighbours to check in either direction.
            lpcnt = 0
            tb = 0
            # -----------------------------------------------------------
            # In single steps check neighbours for available capacity (Start with lower).
            # -----------------------------------------------------------
#                print('')
            while lpcnt < count  and tb < 1: # tb not changed
                lpcnt += 1
#                    print(f'check step {lpcnt}')
                # if exists and not full
                if (bin_tracker['ID']== (b - lpcnt)).any() and bin_tracker['Current'][b - lpcnt] < bin_tracker['Capacity'][b - lpcnt]:
                    tb = b - lpcnt
                    print(f'Found nearest available: {tb}; original target {b} (full/NA)')
                elif (bin_tracker['ID']== (b + lpcnt)).any() and bin_tracker['Current'][b + lpcnt] < bin_tracker['Capacity'][b + lpcnt]:
                    tb = b + lpcnt
                    print(f'Found nearest available: {tb}; original target {b} (full/NA)')
#                else:
#                    validate = False
#                    print(f'MutOpEq failed to find bin for ind. with length {len(ind)}')
#                print(bin_tracker)
#            if (bin_tracker['Capacity'].sum() == bin_tracker['Current'].sum()) and (bin_tracker['Capacity'].sum() == len(population) - 1):
#                tb = b
            
            if tb > 0: # available target bin has been set 
#            print('2 - mutate till it fits')
                while (len(ind) > (tb*bin_width)) or (len(ind) < (bin_width*(tb - 1) + 1)):
                    #if len(ind) more than target bin min: shrink
                    if len(ind) > (tb*bin_width): 
                        gp.mutShrink(ind)
#                            print(f'- shrink: target bin = {tb}, new_length = {len(ind)}') #------------------------monitor
                    #if len(ind) less than target bin min: grow    
                    if len(ind) < (bin_width*(tb - 1) + 1): # target bin max
                        gp.mutInsert(ind, pset)# pset  <------------------------------ ???  
#                            print(f'- grow: target bin = {tb}, new_length = {len(ind)}')  #------------------------monitor
    
    #            evaluate(ind)
    #            if not p1.fitness.valid:
                counteval += 1   #                   <------------------------------ ???    
                xo, yo, zo = toolbox.evaluate(ind, datatrain, datatest) # <------------------------------ ??? 
    #            xo, yo, zo = toolbox.evaluate(p1)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo, 
                #Check if ZeroDivisionError, ValueError i.e. values == (0.0101010101010101,)
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    validate = False
                else: 
                    validate = True
            # No Target bin found    
            else:
                validate = False
                print('Available neighbouring bin not found')

#        except IndexError:    
#            print('Index Error--------------------------------???????')
#            print(f'ind fitness : {ind.fitness.values}')
#            print(f'hof fitness : {halloffame[0].fitness.values}')
#            print(str(ind))
#            exit()

        return validate


#   ###############################################################################        
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                       halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
        
        #+++++++ END OF GENERATION +++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 
#+++++++++++++++++++++++++++++++++++++++++++++
    #Create Report for the Run 
#+++++++++++++++++++++++++++++++++++++++++++++
        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    ## Save 'Hall Of Fame' database
    #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)
    
    
    def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None):
#        nonlocal datatrain, datatest #population, toolbox, report_csv, run, gen, 
        lastgen_db=[]    
        for j in range(len(population)):
            xo, yo, zo = toolbox.evaluate(population[j], datatrain, datatest)
            population[j].fitness.values = xo,
            population[j].evlntime = yo,
            population[j].testfitness = zo,
            lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]), len(population[j]), str(population[j])])
        lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file
        lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(lastgen_csv):
            lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
        else:
            lastgen_dframe.to_csv(lastgen_csv)
#+++++++++++++++++++++++++++++

##------------------------------------------------------------------           
#0000000000000000000000000 NEW GENERATION 00000000000000000000000000000000000000
##=============================================          
##========= Create a new generation ===========
##=============================================
    
    def creategeneration():
        nonlocal population, bin_tracker, popsize, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 
        
        n_accepted = 0
        # reset ['Current'] ---------------------------------------
        bin_tracker['Current']=0
        bin_tracker['Max_fitness']=0
        bin_tracker['Avg_Fitness']=0
        offspring=[]
        
        while n_accepted < popsize:
            ind = breed()
            # identify bin b
            validate = f_validate(ind)
            if validate == True:
                if n_accepted < popsize:
                    offspring.append(ind)
                    n_accepted = n_accepted + 1
                    # Update Bin Tracker --------------------------------------
#                    b = int((len(ind) - 1)/bin_width + 1)
                    b = int( ( ceil(ind.evlntime[0]/unit) - 1)/bin_width + 1 ) #[[[[[[[[[[[[[[[[[[ change ]]]]]]]]]]]]]]]]]]       

                    if (bin_tracker['ID']== b).any():
                      
                        bin_tracker.loc[b,'Current']= bin_tracker['Current'][b] +1
                    else:
                        bin_tracker.loc[b] = [b, 1, 1, 0.0, 0.0]
#                    bin_tracker.loc[b, 'Current'] = bin_tracker.loc[b, 'Current'] + 1 
                    
                    # check if new max for bin and update  -----++++++++++++++++++++++???????????????????????????
                    try:
                        if ind.fitness.values[0] > bin_tracker['Max_fitness'][b]:
                            bin_tracker.loc[b,'Max_fitness'] = ind.fitness.values[0]
#                            print(f'fitness updated for: {b}')                                 
                    except (IndexError):#    except ZeroDivisionError:
                            pass
             
        #======= current population = new population
        population = offspring 
        #========= update halloffame ===============
        #++++++ Update Hof +++++++++++++++++
        updatehof()
        
        #bin_tracker
        updatebintracker()
        updatebincapacity()   

#+++++++++++++++++++++++++++++++++++++++++++++
#Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        creategeneration()
       
        collectStatsGen()
    collectStatsRun()
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++    
    print('collecting stats for the last generation....')
    lastgenstats(population, toolbox, gen=ngen, run=run, report_csv=report_csv)#GEN....??  (9b)
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++

###############################################################################       
    return population, logbook    
###############################################################################
"""    
#==============================================================================
#==============================================================================
"""


def div(left, right):
    return left / right

#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 13), float, "x")
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 5), float, "x") #Airfoil
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #Wine quality
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #Concrete
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 57), float, "x")
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
    with open("/home/aliyu/Documents/CORR/energy_heating/energy_heating.csv") as train:
        trainReader = csv.reader(train)
        Tpoints = list(list(float(item) for item in row) for row in trainReader)
elif system == 'desktop':
    with open("C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Phase3\\Energy_Heating\\energy_heating.csv") as train:
        trainReader = csv.reader(train)
        Tpoints = list(list(float(item) for item in row) for row in trainReader)
elif system == 'laptop':
    with open("C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Phase3\\Energy_Heating\\energy_heating.csv") as train:
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
# toolbox.register("select", tools.selDoubleTournament) # ----------------------------?? breeding
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# When an over the limit child is generated, it is simply replaced by a randomly selected parent.
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.register("worstfitness", tools.selWorst)
#===========================================================     
   
if system == 'server':
	reportfolder = "/home/aliyu/Documents/Target/"
elif system == 'laptop':
	reportfolder = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_TC_OpEq\\"
elif system == 'desktop':
	reportfolder = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_TC_OpEq\\"

"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Initialise -> Get Time equivalent of bin_width
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""
measurements=[]
for r in range(1,16):
    x = 10    #Length of individuals
    x2 = x + bin_width*2
    samplesize = 50
    
    batch1 =[]
    batch2 =[]
    
    while len(batch1) < samplesize:
        sample = toolbox.population(n=1)
        if len(sample[0]) == x:
            batch1.append(sample[0])           
    
    while len(batch2) < samplesize:
        sample = toolbox.population(n=1)
        if len(sample[0]) == x2:
            batch2.append(sample[0])
    
    # Evaluate Batch1 100 times   & capture average   
    #-------------------------------------------------            
    popavgtimes = []
    for p in range(1,51):
        totaltime = 0
        allavgtime = 0
        avgtime = 0
        for k in range(len(batch1)): #Evaluate all and capture evaluation time
            xo, yo, zo = toolbox.evaluate(batch1[k], datatrain, datatest)
            totaltime = totaltime + yo
        avgtime = totaltime/len(batch1)
        popavgtimes.append(avgtime)
    
    firstset = np.mean(popavgtimes)
    print(f'Overall Average of Size {x} : {firstset}')
    #-------------------------------------------------    
    # Evaluate Batch2 100 times   & capture average   
    popavgtimes2 = []
    for p in range(1,51):
        totaltime2 = 0
        allavgtime2 = 0
        avgtime2 = 0
        for k in range(len(batch2)): #Evaluate all and capture evaluation time
            xo, yo, zo = toolbox.evaluate(batch2[k], datatrain, datatest)
            totaltime2 = totaltime2 + yo
        avgtime2 = totaltime2/len(batch2)
        popavgtimes2.append(avgtime2)
    
    secondset = np.mean(popavgtimes2)
    print(f'Overall Average of Size {x2} : {secondset}')
    
    #t_bin_width = round((secondset - firstset)/((x2-x)/bin_width), 15)
    t_bin_width = (secondset - firstset)/((x2-x)/bin_width)
    print(f'bin_width equivalent = {t_bin_width}')
    #-------------------------------------------------
    measurements.append(avgtime2)
print(f'======================')
final = np.median(measurements)
print(f'Final Bin_Equivalent: {final}')

t_bin_width = final

"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""
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

## ================================================
## ================================================
## MutOpEq
## ================================================
### ================================================
#    tag = f'_FL_MutOpEq_Time_P5_EnergyHeat'#Standard GP - Steady State
#    report_csv = f"{reportfolder}{run_time}_{tag}.csv"
#    
#
#    for i in range(1, runs+1):
#    # for i in range(runs+1):
#        run = i
##        pop = toolbox.population(n=20)
#        pop = inipopln()
#        hof = tools.HallOfFame(1) 
#        #-----------------------------------------       
#        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
#        stats_size = tools.Statistics(len)
#        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
#        stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
#        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
#        mstats.register("avg", numpy.mean)
#        mstats.register("std", numpy.std)
#        mstats.register("min", numpy.min)
#        mstats.register("max", numpy.max)
#        pop, log = gpOpEqT(population=pop, toolbox=toolbox, cxpb=0.9, mutpb=0.1, ngen=70, stats=mstats, halloffame=hof, verbose=True,
                                run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, method='MutOpEq', bin_width=5, t_bin_width=t_bin_width, target = 0.16)

# ================================================
# ================================================
# DynOpEq
# ================================================
## ================================================
    tag = f'_FL_DynOpEq_Time_P5_EnergyHeat'#Standard GP - Steady State
    report_csv = f"{reportfolder}{run_time}_{tag}.csv"
 
    for i in range(1, runs+1):
    # for i in range(runs+1):
        run = i
#        pop = toolbox.population(n=20)
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
        pop, log = gpOpEqT(population=pop, toolbox=toolbox, cxpb=0.9, mutpb=0.1, ngen=70, stats=mstats, halloffame=hof, verbose=True,
                                run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, method='DynOpEq', bin_width=5, t_bin_width=t_bin_width, target = 0.16)

#==============================================================================
if __name__ == "__main__":
    main()    
#==============================================================================
