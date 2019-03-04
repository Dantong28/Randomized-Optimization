from __future__ import with_statement

import os
import csv
import time
import sys

from NN_basic import initialize_instances, train

sys.path.append("/Users/dantongzhu/Documents/Spring 2019/Machine Learning/project 2/ABAGAIL/ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

INPUT_LAYER = 12 			#12 inputs for the NN
HIDDEN_LAYER = 10			#10 hidden layers were shown to be optimal in Assignment 1
OUTPUT_LAYER = 1			#one output for the NN

# pop = set of populations
# K = set of ratios for population/tomate and population/tomutate
# if_mut = if do mutation
# iters = number of iterations
def run_GA(pop, K, iters, if_mut):
	n = 5	#Number of trials for each experiemnt, to computer average performance
	instances = initialize_instances()
	factory = BackPropagationNetworkFactory()
	measure = SumOfSquaresError()
	data_set = DataSet(instances)
	for p in pop:
		for k in K:
			tot_corr = 0
			tot_training_time = 0
			tot_testing_time = 0
			for curr_run in range(0, n+1):
				network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
				nnop = NeuralNetworkOptimizationProblem(data_set, network, measure)
				if if_mut:
					oa = StandardGeneticAlgorithm(p, int(p/k), int(p/k), nnop)
				else:
					oa = StandardGeneticAlgorithm(p, int(p/k), 0, nnop)
				#train
				start = time.time()
				train(oa, network, instances, measure, iters)
				end = time.time()
				tot_training_time = tot_training_time + end - start
				optimal_instance = oa.getOptimal()
				network.setWeights(optimal_instance.getData())
				#test
				correct = 0
				incorrect = 0
				start = time.time()
				for instance in instances:
				    network.setInputValues(instance.getData())
				    network.run()
				    predicted = instance.getLabel().getContinuous()
				    actual = network.getOutputValues().get(0)
				    if abs(predicted - actual) < 0.5:
				        correct += 1
				    else:
				        incorrect += 1
				end = time.time()
				tot_testing_time = tot_testing_time + end - start
				tot_corr = tot_corr + (float(correct)/(correct+incorrect))

			training_time = tot_training_time/n
			testing_time = tot_testing_time/n
			corr = tot_corr/n
			print str(corr)
			#print "p = " + str(p) + ", k = " + str(k) + ", " + str(corr)


def main():
	print "GA:"
	iterations = [3, 5, 7, 10, 25, 50, 100, 300, 500, 1000]
	populations = [5, 10, 15, 20, 30, 50, 80, 100, 150, 200]
	K = [1.25, 2, 5, 8, 20]
	#toMate = toMutate = population/k

	Part 1: Fix k = 5, change population
	print "Part 1: Fix k = 5, change population"
	for iters in iterations:
		print "iter = " + str(iters)
		k = 5
		run_GA(populations, [k], iters, True)

	Part 2: Fix p = 100 or 10, change k
	print "GA part 2: Fix p = 100 or 10, change k"
	print "Part 2: Fix p = 100, change k"
	for iters in iterations:
		print "iter = " + str(iters)
		p = 100
		run_GA([p], K, iters, True)

	#Part 3: Fix k = 5 for change to mate, no mutation, varying population
	print "GA part 3: Fix k = 5 for change to mate, no mutation, varying population"
	for iters in iterations:
		print "iter = " + str(iters)
		populations = [3, 5, 8, 10, 15, 20, 30, 50, 80, 100]
		k = 5
		run_GA(populations, [k], iters, False)



if __name__ == "__main__":
    main()
