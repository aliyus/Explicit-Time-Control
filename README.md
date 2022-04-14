# Explicit-Time-Control

The explicit time control uses evaluation time to control the complexity of GP solutions (as introduced in the publications listed below). We use four well-known techniques for bloat-control control evaluation time; hence, the techniques effect an explicit time-control.


# Techniques for Explicit  Time Control
The bloat-control techniques that were adapted to control evaluation time are as follows:

**Death by Size} (DS)** \cite{Luke:2006:EC increases the probability of replacing the larger individuals from the present population. To replace an individual, DS randomly selects two individuals and replaces the larger with a given probability (typically $0.7$; we use the same).
  
    
**Double Tournament} (DT)** \cite{Luke:2006:EC,luke:ppsn2002:pp411} encourages the reproduction of small offspring by increasing the probability of choosing smaller individuals as parents. This is achieved with two rounds of tournaments. In the first round, it runs $n$ probabilistic tournaments each with a tournament of size 2 to select a set of $n$ individuals. Each of these tournaments selects the smaller individual with a probability of $0.7$. Then, in the second round, DT selects the fittest out of the $n$ individuals. %We implemented the DT experiments using steady-state GP.
    
**Operator Equalisation}  (OpEq)** \cite{Dignum:2008:eurogp,journals/gpem/SilvaDV12} allows the sizes of individuals to grow only when fitness is improving. It controls the distribution of size in the population by employing two core functions. The first determines the target distribution (by size) of individuals in the next generation; the second ensures that the next generation matches the target distribution. To define the target distribution, OpEq puts the current individuals into bins according to their sizes and calculates the average fitness score of each bin. This average score is then used to calculate the number of individuals to be allowed in a corresponding bin in the next generation (target distribution). Thus, from one generation to the next the target distribution changes to favour the sizes that produce fitter individuals. In our experiments we used  Dynamic OpEq, which is the variant that produces higher accuracy \cite{journals/gpem/SilvaDV12}.
To adapt OpEq to control evaluation time, we had to estimate the time equivalent of the bin width at the beginning of the run;  we used {\it bin width = 5} in our experiments.
    

**The Tarpeian} (TP)** \cite{poli03} discourages growth in size by penalising larger individuals in the population and making them noncompetitive. This is effected by calculating the average size of the population at every generation and then assigning the worst fitness to a fraction $W$ of the individuals that have above-average size (recommended $W = 0.3$; we use the same). 


Similar to APGP, Death by Size and Double Tournament use Steady-state replacement strategy. However, although Operator Equalisation and Tarpeian use the generational replacement strategy that differs from the steady state in APGP, they are used as benchmarks due to their popularity as bloat control methods. 



# Derived Pulications:

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah.  *“Evolving Simple and Accurate Symbolic Regression Models via Asynchronous Parallel Computing"* In: Applied Soft Computing 104 (2021), p. 107198. ISSN: 1568-4946.
 URL: https://doi.org/10.1016/j.asoc.2021.107198

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah.  *“Time control or size control? reducing complexity and improving the accuracy of genetic programming models"*, In: European Conference on Genetic Programming, Springer, 2020, pp. 195–210. URL: https://doi.org/10.1007/978-3-030-44094-7_13
