
import numpy as np
from EOM_Library.ObjectFunction import *

#---------individual of artificial bee swarm algorithm
class ArtificialBeeIndividual:

    #   vardim: dimension of variables
    #   bound: boundaries of variables
    def __init__(self,  vardim, bound, params):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.trials = 0
        self.position = np.zeros(self.vardim)
        self.params = params

    #   generate a random position for artificial bee swarm algorithm
    def generate(self):
        rnd = np.random.random(size=self.vardim)
        for i in range(0, self.vardim):
            self.position[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    #   calculate the fitness of the position
    def calculateFitness(self):
        self.fitness = MyFitnessFunc(self.vardim, self.position, self.bound,self.params)
