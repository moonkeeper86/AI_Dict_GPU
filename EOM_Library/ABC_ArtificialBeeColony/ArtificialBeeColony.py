import numpy as np
from EOM_Library.ABC_ArtificialBeeColony.ArtificialBeeIndividual import ArtificialBeeIndividual
import random
import copy
import matplotlib.pyplot as plt

#----------the class for artificial bee swarm algorithm
class ArtificialBeeColony:
    #   sizepop: population sizepop
    #   vardim: dimension of variables
    #   bound: boundaries of variables
    #   foodSource： 一只蜜蜂对应一个蜜源，一般情况下雇佣蜂和观察蜂数量各为蜂群数量一般
    #   MAXGEN: termination condition，最多迭代次数
    #   params: algorithm required parameters, it is a list which is consisting of[trailLimit, C]
    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.foodSource = int(self.sizepop / 2)
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.foodSource, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
    #   initialize the population of abs
    def initialize(self):
        for i in range(0, int(self.foodSource)):
            ind = ArtificialBeeIndividual(self.vardim, self.bound,self.params)
            ind.generate()
            self.population.append(ind)
    #   evaluation the fitness of the population
    def evaluation(self):
        for i in range(0, int(self.foodSource)):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
    #   employed bee phase
    def employedBeePhase(self):
        for i in range(0, int(self.foodSource)):                      #对每一个蜜源
            k = np.random.random_integers(0, self.vardim - 1)         #找出解的随机维
            j = np.random.random_integers(0, self.foodSource - 1)     #找出随机一个蜜源
            while j == i:
                j = np.random.random_integers(0, self.foodSource - 1)
            vi = copy.deepcopy(self.population[i])                    #选定当前蜜源的雇佣蜂
            # 当前蜜蜂与随机选定的蜜源的第k维做随机更新，作为预备更新
            vi.position[k] += np.random.uniform(low=-1, high=1.0, size=1) * (vi.position[k] - self.population[j].position[k])
            # 更新位置需要满足边界条件
            if vi.position[k] < self.bound[0, k]:
                vi.position[k] = self.bound[0, k]
            if vi.position[k] > self.bound[1, k]:
                vi.position[k] = self.bound[1, k]
            # 对当前蜜蜂更新fitness值
            vi.calculateFitness()
            # 判断fitness值是否有提升
            if vi.fitness > self.fitness[i]:
                self.population[i] = vi        # fitness提升则用现有更新位置指挥蜜蜂前往
                self.fitness[i] = vi.fitness   # 确定更新fitness
                if vi.fitness > self.best.fitness:  #判断当前蜜蜂行为是否表现最好
                    self.best = vi
            else:
                self.population[i].trials += 1
    #   onlooker bee phase
    def onlookerBeePhase(self):
        accuFitness = np.zeros((int(self.foodSource), 1))
        maxFitness = np.max(self.fitness)

        for i in range(0, int(self.foodSource)):
            accuFitness[i] = 0.9 * self.fitness[i] / maxFitness + 0.1

        for i in range(0, int(self.foodSource)):
            for fi in range(0, int(self.foodSource)):
                r = random.random()
                if r < accuFitness[i]:
                    k = np.random.random_integers(0, self.vardim - 1)
                    j = np.random.random_integers(0, self.foodSource - 1)
                    while j == fi:
                        j = np.random.random_integers(0, self.foodSource - 1)
                    vi = copy.deepcopy(self.population[fi])
                    vi.position[k] += np.random.uniform(low=-1, high=1.0, size=1) * (vi.position[k] - self.population[j].position[k])
                    if vi.position[k] < self.bound[0, k]:
                        vi.position[k] = self.bound[0, k]
                    if vi.position[k] > self.bound[1, k]:
                        vi.position[k] = self.bound[1, k]
                    vi.calculateFitness()
                    if vi.fitness > self.fitness[fi]:
                        self.population[fi] = vi
                        self.fitness[fi] = vi.fitness
                        if vi.fitness > self.best.fitness:
                            self.best = vi
                    else:
                        self.population[fi].trials += 1
                    break
    #   scout bee phase
    def scoutBeePhase(self):
        for i in range(0, int(self.foodSource)):
            if self.population[i].trials > self.params[0]:
                self.population[i].generate()
                self.population[i].trials = 0
                self.population[i].calculateFitness()
                self.fitness[i] = self.population[i].fitness
    #   the evolution process of the abs algorithm
    def solve(self):
        self.times = 0       #当前问题求解的迭代次数号码
        self.initialize()
        self.evaluation()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])   #选出当前最好的蜜蜂
        self.avefitness = np.mean(self.fitness)
        self.trace[self.times, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.times, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.times, self.best.fitness, np.mean(self.fitness)))
        while self.times < self.MAXGEN - 1:
            self.times += 1
            self.employedBeePhase()
            self.onlookerBeePhase()
            self.scoutBeePhase()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.times, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.times, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.times, self.best.fitness, np.mean(self.fitness)))
        print("Optimal function value is: %f; " % self.trace[self.times, 0])
        print("Optimal solution is:")
        print(self.best.position)
        self.printResult()
    #   plot the result of abs algorithm
    def printResult(self):
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial Bee Swarm algorithm for function optimization")
        plt.legend()
        plt.show()