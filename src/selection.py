import numpy as np
import random
from operator import attrgetter

"""
There are three different Selection methods:
1. Roulette
2. Tournament
3. Lexicase
"""

class Roulette:
    def __init__(self, args, pop):
        self.probabilities = np.array([pop.highest_fitness - ind.fitness for ind in pop.pop])
        self.probabilities /= np.sum(self.probabilities)
        self.k = args['k']

    # Returns list of Individuals selected
    def select(self, pop):
        selected = []
        for _ in range(self.k):
            idx = np.random.choice(np.arange(pop.size), p=self.probabilities)
            selected.append(pop.pop[idx])
        return selected
 

class Tournament:
    def __init__(self, args, pop):
        self.k = args['k']

    # Returns list of 2 Individuals selected by tournamnet
    def select(self, pop):
        selected = random.sample(pop.pop, self.k)
        dad = min(selected, key=attrgetter('fitness'))
        selected = random.sample(pop.pop, self.k)
        mom = min(selected, key=attrgetter('fitness'))
        return [dad, mom]
 
 
class Lexicase:
    def select(self, pop):
        pass
 

def SelectionFactory(selection_args, population):
    methods = {
        'roulette': Roulette,
        'tournament': Tournament,
        'lexicase': Lexicase,
    }
    return methods[selection_args['name']](selection_args['args'], population)