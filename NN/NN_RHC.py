from __future__ import with_statement

import os
import csv
import time
import sys

from NN_basic import initialize_instances, train

sys.path.append("/Users/dantongzhu/Documents/Spring 2019/Machine Learning/ABAGAIL/ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

INPUT_LAYER = 12 			#12 inputs for the NN
HIDDEN_LAYER = 10			#10 hidden layers were shown to be optimal in Assignment 1
OUTPUT_LAYER = 1			#one output for the NN


def main():
	instances = initialize_instances()
	factory = BackPropagationNetworkFactory()
	measure = SumOfSquaresError()
	data_set = DataSet(instances)

	iterations = [5, 10, 20, 30, 50, 70, 100, 200, 300, 500, 800, 1000]
	#iterations = [1, 3, 5, 7, 10, 20, 30, 50, 70, 100]
	n = 5
	print "RHC:"
	for iters in iterations:

		tot_corr = 0
		tot_training_time = 0
		tot_testing_time = 0

		for curr_run in range(0, n+1):
			network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
			nnop = NeuralNetworkOptimizationProblem(data_set, network, measure)
			oa = RandomizedHillClimbing(nnop)

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
		print str(training_time) + "\n"
		#print "iter = " + str(iters) + ", " + str(corr)

		#print "correctness = " + str(corr)
		#print "training time = " + str(training_time)
		#print "testing time = " + str(testing_time)


if __name__ == "__main__":
    main()

