
from deap import tools
from multiprocessing.pool import ThreadPool, threading
import random
from operator import add, itemgetter
import numpy as np
import pandas as pd
import os
from functools import reduce


def apgpwGenStats(population, toolbox, cxpb, mutpb, ngen, poolsize, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None ):
     
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
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
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
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
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
        
#+++++++++++++++++++++++++++++++++++++++++++++
#Breeding Function
#+++++++++++++++++++++++++++++++++++++++++++++
# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, poolsize

        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

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
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            counteval_lock.release()
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
#                print('check this out')
#                print(str(p1))
#                print(str(p1.fitness.values))

        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ################################################################################        
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        #=============HOF Evaluation Time ==========================
        #Capture Evaluation Time of HOF outside the threading
        xo, yo, zo = toolbox.evaluate(halloffame[0], datatrain, datatest)
        halloffame[0].evlntime = yo,
        #===========================================================
                
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
        
#    #=============HOF Evaluation Time ==========================
#    #Capture Evaluation Time of HOF outside the threading
#        for j in range(len(halloffame)):
#            xo, yo, zo = toolbox.evaluate(halloffame[j], datatrain, datatest)
#            halloffame[j].evlntime = yo,
#    #===========================================================
#    
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

#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++CORRELATION+++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
#        corrln = hof_dframe['Length'].corr(hof_dframe['Evln_time'])       
#        print('APGP - Correlation btw Evaluation Time and Size: ', corrln)
#        print(f'Run: {Run}, Generation: {Generation} ', corrln)

#===========================================================
    #Function to collect stats for the last generation
    def checkcorr():
    #def checkcorr(population, toolbox, gen,  run, report_csv):
        nonlocal population, toolbox, report_csv, run, gen
        checkcorr_db=[]    
        checkcorr2_db=[]            
        for j in range(len(population)):
            xo, yo, zo = toolbox.evaluate(population[j], datatrain, datatest)
    #        population[j].fitness.values = xo,
            population[j].evlntime = yo,
    #        population[j].testfitness = zo,
            checkcorr_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]), len(population[j])])
        checkcorr_dframe=pd.DataFrame(checkcorr_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length'])
    
        #--------------------------------------------------------------------------
        #Destination file
        checkcorr_csv = f'{report_csv[:-4]}_corr.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(checkcorr_csv):
            checkcorr_dframe.to_csv(checkcorr_csv, mode='a', header=False)
        else:
            checkcorr_dframe.to_csv(checkcorr_csv)
        #--------------------------------------------------------------------------
        # Correlation
        corrln = checkcorr_dframe['Length'].corr(checkcorr_dframe['Evln_time'])       
        print('APGP - Correlation btw Evaluation Time and Size: ', corrln)
        #Destination file

        checkcorr2_db.append([run, gen, corrln])
        checkcorr2_dframe=pd.DataFrame(checkcorr2_db, columns=['Run', 'Generation', 'Correlation_size-time'])

        checkcorr2_csv = f'{report_csv[:-4]}_2corr.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(checkcorr2_csv):
            checkcorr2_dframe.to_csv(checkcorr2_csv, mode='a', header=False)
        else:
            checkcorr2_dframe.to_csv(checkcorr2_csv)            
    #===========================================================
    #===========================================================  
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
        
#+++++++++++++++++++++++++++++++++++++++++++++
#Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
#        tp = Pool(processes=4)    
        tp = ThreadPool(poolsize)  # <-------------------------------------------- (3a)
        # Generate offsprings -  equivalent to a generation based on populations size
        poplnsize =  len(population)
#        poplnsize =  500
        counteval = 0 
#        while (counteval < poplnsize+1):  
        for h in range(poplnsize):
#            print(counteval)
            tp.apply_async(breed)
#   Append the current generation statistics to the logbook
        print(threading.active_count())
        tp.close()#<------------------------------------------------???????????
        tp.join()#  <-----------------------------------------------???????????     
        
        while counteval < poplnsize:
#            print(f'more pending')
            tp = ThreadPool(poolsize)  # <---------------------------------------- (3b)
#if count < target:psize
            for j in range(poplnsize - counteval):
                tp.apply_async(breed)
            tp.close() # <---------------------------------------??????????????
            tp.join() #  <---------------------------------------??????????????
#            print(f'done: {counteval}')        
        collectStatsGen()
        # checkcorr() #  <---------------------------------------check correlation ========???
#        print(threading.active_count())
    collectStatsRun()
    
###############################################################################
###############################################################################       
    return population, logbook    
###############################################################################
"""    
#==============================================================================
#==============================================================================
"""






"""
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
# Standard GP - Steady State
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
"""
def gpSteadyState(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
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
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
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
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
#+++++++++++++++++++++++++++++++++++++++++++++
#Breeding Function
#+++++++++++++++++++++++++++++++++++++++++++++
# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock

        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

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
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            counteval_lock.release()
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
#                print('check this out')
#                print(str(p1))
#                print(str(p1.fitness.values))
        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
#        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
#        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ################################################################################        
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

#+++++++++++++++++++++++++++++++++++++++++++++
#Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        counteval = 0 
        for h in range(poplnsize):
            breed()
        
        while counteval < poplnsize:
#            print(f'more pending')
            for j in range(poplnsize - counteval):
                breed()

        collectStatsGen()
    collectStatsRun()
    
###############################################################################       
    return population, logbook    
###############################################################################
"""    
#==============================================================================
#==============================================================================
"""


"""
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
# Double Tournament GP - Steady State
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
"""
def gpDoubleT(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
	
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    #target = 0.02 							#--------------------------------------------------------------(())
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````  


    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
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

    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
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
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
#+++++++++++++++++++++++++++++++++++++++++++++
#Breeding Function
#+++++++++++++++++++++++++++++++++++++++++++++
# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, mettarget

        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        """
        Select parents using double-tournament for bloat control.
        """
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=2, fitness_first=True)))
#        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
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
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            counteval_lock.release()
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
#                print('check this out')
#                print(str(p1))
#                print(str(p1.fitness.values))
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
			#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
            # if float(p1.testfitness[0]) >= target:
            if float(p1.fitness.values[0]) >= target:
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




        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
#        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
#        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ################################################################################        
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

#+++++++++++++++++++++++++++++++++++++++++++++
#Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        counteval = 0 
        for h in range(poplnsize):
            breed()
        
        while counteval < poplnsize:
#            print(f'more pending')
            for j in range(poplnsize - counteval):
                breed()

        collectStatsGen()
    collectStatsRun()
    
###############################################################################       
    return population, logbook    
###############################################################################
"""    
#==============================================================================
#==============================================================================
"""



"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The APGP_No_Generational_Stats - Parallel Steady State GP
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
def apgpNoGenStats(population, toolbox, cxpb, mutpb, ngen, poolsize, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None ):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    
    # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
    # get probability of evaluations and factor it in the number of breeds initiated
    # This will allow race to continue without stopping to check.
    factor = 1/((cxpb + mutpb) - cxpb*mutpb)
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````  
    
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
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
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
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

    # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
    gen=70
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````
    
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
        
#+++++++++++++++++++++++++++++++++++++++++++++
#Breeding Function
#+++++++++++++++++++++++++++++++++++++++++++++
# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, poolsize

        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

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
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            
            # # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
            # if counteval % poplnsize == 0:
                # print(f'{counteval} evaluations initiated -- 	{round((100*counteval)/(ngen*poplnsize),2)}% of run {run}')
            # # ````````````````````````````````````````````````````````````````````````````````````````````````````````

            counteval_lock.release()
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
#                print('check this out')
#                print(str(p1))
#                print(str(p1.fitness.values))

        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ################################################################################        
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        #=============HOF Evaluation Time ==========================
        #Capture Evaluation Time of HOF outside the threading
        xo, yo, zo = toolbox.evaluate(halloffame[0], datatrain, datatest)
        halloffame[0].evlntime = yo,
        #===========================================================
                
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
        
#    #=============HOF Evaluation Time ==========================
#    #Capture Evaluation Time of HOF outside the threading
#        for j in range(len(halloffame)):
#            xo, yo, zo = toolbox.evaluate(halloffame[j], datatrain, datatest)
#            halloffame[j].evlntime = yo,
#    #===========================================================
#    
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


# NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
##+++++++++++++++++++++++++++++++++++++++++++++
##Create a Generation
##+++++++++++++++++++++++++++++++++++++++++++++
#    # Begin the generational process   
#    for gen in range(1, ngen+1):

    tp = ThreadPool(poolsize)  # <-------------------------------------------- (3a)
        # Generate offsprings -  equivalent to a generation based on populations size
    poplnsize =  len(population)
    targetevalns = poplnsize*ngen
#        poplnsize =  500
    
    counteval = 0 
#        while (counteval < poplnsize+1):  
    for h in range(int(poplnsize*ngen*factor)):
#            print(counteval)
        tp.apply_async(breed)

#   Append the current generation statistics to the logbook
    tp.close() # <---------------------------------------??????????????
    tp.join() #  <---------------------------------------??????????????

    print(f'done  : {counteval}')
    print(f'Target: {targetevalns}')
    print(threading.active_count())

	
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''	<--------------------????????????????????????????
	# Re Evaluate the final population outside the thread
    for j in range(len(population)):
        xo, yo, zo = toolbox.evaluate(population[j], datatrain, datatest)
        population[j].fitness.values = xo,
        population[j].evlntime = yo,
        population[j].testfitness = zo,
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
#    tp = ThreadPool(poolsize)  # <-------------------------------------------- (3a)
##    while counteval < poplnsize*ngen:
#    for j in range(targetevalns - counteval):
#        tp.apply_async(breed)
#    tp.close() # <---------------------------------------??????????????
#    tp.join() #  <---------------------------------------??????????????
#    
#    print(f'done  : {counteval}')
#    print(f'Target: {targetevalns}')
#    print(f'Threads running: {threading.active_count()}')
    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    collectStatsGen()
    collectStatsRun()
# `````````````````````````````````````````````````````````````````````````````   
    
###############################################################################
###############################################################################       
    return population, logbook    
###############################################################################
"""    
#==============================================================================
#==============================================================================
"""









#===========================================================
#======== Collect Stats for the Final Generation ===========
#===========================================================
#===========================================================
#Function to collect stats for the last generation
def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None, datatrain=None, datatest=None):
#    nonlocal population, toolbox, report_csv, run, gen
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
#===========================================================
#===========================================================
#===========================================================







