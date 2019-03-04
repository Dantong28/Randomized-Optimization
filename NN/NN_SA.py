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
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

INPUT_LAYER = 12 			#12 inputs for the NN
HIDDEN_LAYER = 10			#10 hidden layers were shown to be optimal in Assignment 1
OUTPUT_LAYER = 1			#one output for the NN

#oa = SimulatedAnnealing(1E11, .95, nnop)

# T_set = set of starting temperatures
# c_set = set of cooling exponents
# iters = number of iterations
def run_SA(T_set, c_set, iters):
	n = 5	#Number of trials for each experiemnt, to computer average performance
	instances = initialize_instances()
	factory = BackPropagationNetworkFactory()
	measure = SumOfSquaresError()
	data_set = DataSet(instances)

	for T in T_set:
		for c in c_set:
			tot_corr = 0
			tot_training_time = 0
			tot_testing_time = 0

			for curr_run in range(0, n+1):
				network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
				nnop = NeuralNetworkOptimizationProblem(data_set, network, measure)
				oa = SimulatedAnnealing(T, c, nnop)
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
			#print "T = " + str(T) + ", c = " + str(c) + ", " + str(corr)



def main():

	iterations = [5, 10, 20, 30, 50, 70, 100, 200, 300, 500, 800, 1000, 1500, 2000]
	#Part 1: fixed starting T with varying cooling temperature
	T_set = [1E11]
	c_set = [0.1, 0.25, 0.4, 0.55, 0.70, 0.85, 0.95]
	print "SA:"
	for iters in iterations:
		print "iters = " + str(iters)
		run_SA(T_set, c_set, iters)


if __name__ == "__main__":
    main()
