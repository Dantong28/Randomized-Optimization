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

	iterations = [5, 10, 25, 50, 100, 300, 500, 1000]
	n = 5
	for iters in iterations:
		print "iters = " + str(iters)
		networks = []  # BackPropagationNetwork
		nnop = []  # NeuralNetworkOptimizationProblem
		oa = []  # OptimizationAlgorithm
		oa_names = ["RHC", "SA", "GA"]

		tot_RHC_corr = 0
		tot_RHC_training_time = 0
		tot_SA_corr = 0
		tot_SA_training_time = 0
		tot_GA_corr = 0
		tot_GA_training_time = 0

		for curr_run in range(0, n+1):
			for name in oa_names:
				classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
				networks.append(classification_network)
				nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))
			oa.append(RandomizedHillClimbing(nnop[0]))
			oa.append(SimulatedAnnealing(1E11, .2, nnop[1]))
			oa.append(StandardGeneticAlgorithm(50, 10, 10, nnop[2]))

			for i, name in enumerate(oa_names):
				#train
				start = time.time()
				train(oa[i], networks[i], instances, measure, iters)
				end = time.time()
				if i == 0:
					tot_RHC_training_time = tot_RHC_training_time + end - start
				if i == 1:
					tot_SA_training_time = tot_SA_training_time + end - start
				if i == 2:
					tot_GA_training_time = tot_GA_training_time + end - start

				optimal_instance = oa[i].getOptimal()
				networks[i].setWeights(optimal_instance.getData())
				#test
				correct = 0
				incorrect = 0
				for instance in instances:
				    networks[i].setInputValues(instance.getData())
				    networks[i].run()
				    predicted = instance.getLabel().getContinuous()
				    actual = networks[i].getOutputValues().get(0)
				    if abs(predicted - actual) < 0.5:
				        correct += 1
				    else:
				        incorrect += 1
				end = time.time()

				if i == 0:
					tot_RHC_corr = tot_RHC_corr + (float(correct)/(correct+incorrect))
				if i == 1:
					tot_SA_corr = tot_SA_corr + (float(correct)/(correct+incorrect))
				if i == 2:
					tot_GA_corr = tot_GA_corr + (float(correct)/(correct+incorrect))
		RHC_training_time = tot_RHC_training_time/n
		SA_training_time = tot_SA_training_time/n
		GA_training_time = tot_GA_training_time/n
		RHC_corr = tot_RHC_corr/n
		SA_corr = tot_SA_corr/n
		GA_corr = tot_GA_corr/n
		print "corr:"
		print str(RHC_corr)
		print str(SA_corr)
		print str(GA_corr)
		print "time:"
		print str(RHC_training_time)
		print str(SA_training_time)
		print str(GA_training_time) + "\n"


if __name__ == "__main__":
    main()


# def main():
# 	iterations = [3, 5, 7, 10, 25, 50, 100, 300, 500, 1000]

#     """Run algorithms on the abalone dataset."""
#     instances = initialize_instances()
#     factory = BackPropagationNetworkFactory()
#     measure = SumOfSquaresError()
#     data_set = DataSet(instances)

#     networks = []  # BackPropagationNetwork
#     nnop = []  # NeuralNetworkOptimizationProblem
#     oa = []  # OptimizationAlgorithm
#     oa_names = ["RHC", "SA", "GA"]
#     results = ""

#     for name in oa_names:
#         classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
#         networks.append(classification_network)
#         nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

#     oa.append(RandomizedHillClimbing(nnop[0]))
#     oa.append(SimulatedAnnealing(1E11, .2, nnop[1]))
#     oa.append(StandardGeneticAlgorithm(50, 10, 10, nnop[2]))

#     for i, name in enumerate(oa_names):
#         start = time.time()
#         correct = 0
#         incorrect = 0

#         train(oa[i], networks[i], oa_names[i], instances, measure)
#         end = time.time()
#         training_time = end - start

#         optimal_instance = oa[i].getOptimal()
#         networks[i].setWeights(optimal_instance.getData())

#         start = time.time()
#         for instance in instances:
#             networks[i].setInputValues(instance.getData())
#             networks[i].run()

#             predicted = instance.getLabel().getContinuous()
#             actual = networks[i].getOutputValues().get(0)

#             if abs(predicted - actual) < 0.5:
#                 correct += 1
#             else:
#                 incorrect += 1

#         end = time.time()
#         testing_time = end - start

#         results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
#         results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
#         results += "\nTraining time: %0.03f seconds" % (training_time,)
#         results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

#     print results

