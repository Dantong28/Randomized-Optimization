import sys
import os
import time

sys.path.append("/Users/dantongzhu/Documents/Spring 2019/Machine Learning/project 2/ABAGAIL/ABAGAIL.jar")
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
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

from array import array



"""
Commandline parameter(s):
   none
"""

N=200
T=N/5
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


#Part 1: compare 4 algorithms
print "Part 1: compare 4 algorithms"
iterations = [50, 100, 200, 300, 500, 700, 1000, 3000, 4500, 6000, 7500, 9000, 10000]
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
		tot_eval_rhc = tot_eval_rhc + ef.value(rhc.getOptimal())
		end = time.time()
		tot_time_rhc = tot_time_rhc + end - start

		#SA
		start = time.time()
		sa = SimulatedAnnealing(1E11, .95, hcp)
		fit = FixedIterationTrainer(sa, iters)
		fit.train()
		tot_eval_sa = tot_eval_sa + ef.value(sa.getOptimal())
		end = time.time()
		tot_time_sa = tot_time_sa + end - start

		#GA
		start = time.time()
		ga = StandardGeneticAlgorithm(200, 100, 10, gap)
		fit = FixedIterationTrainer(ga, iters)
		fit.train()
		tot_eval_ga = tot_eval_ga + ef.value(ga.getOptimal())
		end = time.time()
		tot_time_ga = tot_time_ga + end - start

		#MIMIC
		start = time.time()
		mimic = MIMIC(200, 20, pop)
		fit = FixedIterationTrainer(mimic, iters)
		fit.train()
		tot_eval_mimic = tot_eval_mimic + ef.value(mimic.getOptimal())
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

# #Part 2: More iterations for MIMIC
# print "Part 2: More iterations for MIMIC"
# iterations = [20000, 50000, 80000, 120000]
# for iters in iterations:
# 	mimic = MIMIC(200, 20, pop)
# 	fit = FixedIterationTrainer(mimic, iters)
# 	fit.train()
# 	print "iters = " + str(iters) + ", " + str(ef.value(mimic.getOptimal()))

#Result:
# iters = 20000, 200.0
# iters = 50000, 200.0
# iters = 80000, 200.0
# iters = 120000, 200.0



