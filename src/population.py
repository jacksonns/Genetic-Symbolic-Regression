from src.genotype import Genotype
from math import sin, cos, log, sqrt
import numpy as np
import random
from collections import Counter

class Individual:
    def __init__(self, genotype=None):
        if not genotype:
            self.genotype = Genotype('full')
        else:
            self.genotype = genotype
        self.tree = self.genotype.generate_tree()
        self.fitness = None

    def get_str_expression(self):
        return self.tree.get_str_expression()
    
    # Root Mean square error as fitness 
    def set_RMSE(self, train_data):
        expr = self.get_str_expression()
        size = np.size(train_data.X)
        Fxy_ = np.zeros(size)
        for i in range(size):
            x = train_data.X[i]
            y = train_data.Y[i]
            try:
                Fxy_[i] = eval(expr)
            except:
                Fxy_[i] = 0
        np.nan_to_num(Fxy_, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        sum = np.sum(np.power((Fxy_ - train_data.Fxy), 2))
        mean = np.mean(train_data.Fxy)
        norm = np.sum(np.power((train_data.Fxy - mean), 2))
        self.fitness = np.sqrt(sum / norm)
    
    def get_RMSE(self, train_data):
        if not self.fitness:
            self.set_RMSE(train_data)
        return self.fitness
    
    def crossover(self, partner):
        return self.genotype.crossover(partner.genotype)
    
    def mutation(self):
        return self.genotype.mutation()


class Population:
    def __init__(self, population_size=0):
        self.size = population_size
        self.pop = []
        if self.size > 0:
            self.init_population()
        self.lowest_fitness = None
        self.highest_fitness = None
        self.sum_fitness = 0
        self.mean_fitness= 0
        self.best_ind = None
    
    def init_population(self):
        half = self.size // 2
        for _ in range(half):
            gen = Genotype('grow')
            self.pop.append(Individual(gen))
        for _ in range(self.size - half):
            gen = Genotype('full')
            self.pop.append(Individual(gen))
        random.shuffle(self.pop)
    
    def add_individuals(self, individuals):
        for ind in individuals:
            self.pop.append(ind)
            self.size += 1
    
    def remove_last_individual(self):
        del self.pop[-1]
        self.size -= 1

    def count_unique_individuals(self):
        counter = Counter(ind.fitness for ind in self.pop)
        return len(counter)
    
    def evaluate(self, train_data):
        for ind in self.pop:
            fitness = ind.get_RMSE(train_data)
            self.sum_fitness += fitness
            if not self.lowest_fitness and not self.highest_fitness:
                self.lowest_fitness = fitness
                self.best_ind = ind
                self.highest_fitness = fitness
            if fitness < self.lowest_fitness:
                self.lowest_fitness = fitness
                self.best_ind = ind
            if fitness > self.highest_fitness:
                self.highest_fitness = fitness
        self.mean_fitness = self.sum_fitness / len(self.pop)
            
