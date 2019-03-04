"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.
Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time
import sys

sys.path.append("/Users/dantongzhu/Documents/Spring 2019/Machine Learning/project 2/ABAGAIL/ABAGAIL.jar")

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm


INPUT_FILE_NAME = "forestfires.csv"


def month_day_to_number(string):
    m = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12,
         'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
    s = string.strip()[:3].lower()
    return m[s]

def initialize_instances():
    instances = []
    with open(INPUT_FILE_NAME, "r") as forestfires:
        reader = csv.reader(forestfires)
        for row in reader:
            #instance = Instance([float(value) for value in row[:-1]])
            a = []
            for i in range(0, len(row)-1):
                if row[i].isalpha():
                    num = month_day_to_number(row[i])
                else:
                    num = float(row[i])
                a.append(num)
            instance = Instance(a)

            if float(row[-1]) < 5:
                num = 0
            else:
                num = 1
            instance.setLabel(Instance(num))
            instances.append(instance)

    return instances

def train(oa, network, instances, measure, iters):
    for iteration in xrange(iters):
        oa.train()
        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()
            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)
