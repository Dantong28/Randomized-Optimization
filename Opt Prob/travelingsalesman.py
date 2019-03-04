# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
import sys
import os
import time

sys.path.append("/Users/dantongzhu/Documents/Spring 2019/Machine Learning/project 2/ABAGAIL/ABAGAIL.jar")

import shared.Instance as inst

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array




"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 50
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]

for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
    #print (points[i][0], points[i][1])

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# for mimic we use a sort encoding
ef_mimic = TravelingSalesmanSortEvaluationFunction(points);
fill = [N] * N
ranges = array('i', fill)
odd = DiscreteUniformDistribution(ranges);
df = DiscreteDependencyTree(.1, ranges); 
pop = GenericProbabilisticOptimizationProblem(ef_mimic, odd, df);

# "Inverse of Distance"
iterations = [10, 20, 30, 40, 50, 100, 200, 300, 500, 700, 1000, 5000, 10000]
n = 5

for iters in iterations:
	print "iters = " + str(iters)
	tot_eval_rhc = 0
	tot_eval_sa = 0
	tot_eval_ga = 0
	tot_eval_mimic = 0

	tot_time_rhc = 0
	tot_time_sa = 0
	tot_time_ga = 0
	tot_time_mimic = 0

   	for i in range(1, n+1):

		#RHC
		start = time.time()
		rhc = RandomizedHillClimbing(hcp)
		fit = FixedIterationTrainer(rhc, iters)
		fit.train()
		tot_eval_rhc = tot_eval_rhc + 1/ef.value(rhc.getOptimal())
		end = time.time()
		tot_time_rhc = tot_time_rhc + end - start

		#SA
		start = time.time()
		sa = SimulatedAnnealing(1E12, .999, hcp)
		fit = FixedIterationTrainer(sa, iters)
		fit.train()
		tot_eval_sa = tot_eval_sa + 1/ef.value(sa.getOptimal())
		end = time.time()
		tot_time_sa = tot_time_sa + end - start

		#GA
		start = time.time()
		ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
		fit = FixedIterationTrainer(ga, iters)
		fit.train()
		tot_eval_ga = tot_eval_ga + 1/ef.value(ga.getOptimal())
		end = time.time()
		tot_time_ga = tot_time_ga + end - start

		#MIMIC
		start = time.time()
		mimic = MIMIC(500, 100, pop)
		fit = FixedIterationTrainer(mimic, iters)
		fit.train()
		tot_eval_mimic = tot_eval_mimic + 1/ef_mimic.value(mimic.getOptimal())
		end = time.time()
		tot_time_mimic = tot_time_mimic + end - start

	eval_rhc = tot_eval_rhc/n
	eval_sa = tot_eval_sa/n
	eval_ga = tot_eval_ga/n
	eval_mimic = tot_eval_mimic/n

	time_rhc = 1000*tot_time_rhc/n
	time_sa = 1000*tot_time_sa/n
	time_ga = 1000*tot_time_ga/n
	time_mimic = 1000*tot_time_mimic/n

	print "evaluation:"
	print str(eval_rhc)
	print str(eval_sa)
	print str(eval_ga)
	print str(eval_mimic)
	print "time:"
	print str(time_rhc)
	print str(time_sa)
	print str(time_ga)
	print str(time_mimic)
	print ""






	

