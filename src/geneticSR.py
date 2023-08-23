from src.population import Individual, Population
from src.selection import SelectionFactory
import numpy as np
import pandas as pd
from operator import attrgetter
from copy import deepcopy
import os


class TrainData:
    def __init__(self, train_file):
        df = pd.read_csv(train_file, header=None)
        self.X = df[0].to_numpy()
        self.Y = df[1].to_numpy()
        self.Fxy = df[2].to_numpy()


class IOData:
    def __init__(self, generations):
        # Each list has the mean value for 30 executions for each generation
        self.lowest_fitness = np.zeros(generations)
        self.mean_fitness = np.zeros(generations)
        self.unique_ind = np.zeros(generations)
        self.better_children = np.zeros(generations)
        self.best_ind = None
    
    def get_data(self, generation, pop):
        self.lowest_fitness[generation] += pop.lowest_fitness
        self.mean_fitness[generation] += pop.mean_fitness
        self.unique_ind[generation] += pop.count_unique_individuals()
        if not self.best_ind:
            self.best_ind = pop.best_ind
        else:
            if pop.best_ind.fitness < self.best_ind.fitness:
                self.best_ind = pop.best_ind
    
    def calculate_mean(self, exec_num):
        self.lowest_fitness = self.lowest_fitness / exec_num
        self.mean_fitness = self.mean_fitness / exec_num
        self.unique_ind = self.unique_ind / exec_num
        self.better_children = self.better_children / exec_num

    def load_results(self, results_path):
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.lowest_fitness.tofile(results_path+'/lowest_fitness.csv', sep=',')
        self.mean_fitness.tofile(results_path+'/mean_fitness.csv', sep=',')
        self.unique_ind.tofile(results_path+'/unique_ind.csv', sep=',')
        self.better_children.tofile(results_path+'/better_children.csv', sep=',')
        with open(results_path+'/best_ind.csv', 'w') as f:
            f.write('{},{},0'.format(self.best_ind.fitness, 
                                      self.best_ind.get_str_expression()))
        

# Genetic Symbolic Regressor
class GeneticSR:
    def __init__(self, args):
        self.population_size = args['population_size'] 
        self.generations = args['generations']
        self.selection = args['selection']
        self.mutation_prob = args['mutation_prob']
        self.crossover_prob = args['crossover_prob']
        self.elitism = args['elitism']
        self.verbose = args['verbose']
        self.train_data = TrainData(args['train_file'])
        self.io = IOData(self.generations)

    def genetic_evolve(self, generation, ind):
        # Define if performs mutation (0) or crossover (1) or (2) keep individuals
        choice = np.random.choice(np.array([0, 1, 2]), p=[self.mutation_prob, self.crossover_prob, 
                                                          1-self.mutation_prob-self.crossover_prob])
        mean_parents = (ind[0].fitness + ind[1].fitness) / 2
        if choice == 1:
            offspring = ind[0].crossover(ind[1])
            child1 = Individual(offspring[0])
            child2 = Individual(offspring[1])
            child1.set_RMSE(self.train_data)
            child2.set_RMSE(self.train_data)

            if child1.fitness == child2.fitness:
                return[ind[0], child1]

            if child1.fitness < mean_parents:
                self.io.better_children[generation] += 1
            if child2.fitness < mean_parents:
                self.io.better_children[generation] += 1

            options = [ind[0], ind[1], child1, child2]
            options = sorted(options, key=attrgetter('fitness'))
            return [options[0], options[1]]
        
        elif choice == 0:
            i = np.random.randint(len(ind))
            offspring = ind[i].mutation()
            child = Individual(offspring)

            child.set_RMSE(self.train_data)
            if child.fitness < ind[i].fitness:
                self.io.better_children[generation] += 1
                #return [child]
            
            return [child]
        else: return ind

    def run(self):
        pop = Population(self.population_size)
        for i in range(self.generations):
            pop.evaluate(self.train_data)
            sel = SelectionFactory(self.selection, pop)
            new_pop = Population()

            if self.verbose:
                print('Generation {}:'.format(i))
                print('Unique individuals: {}'.format(pop.count_unique_individuals()))
                print('Lowest: {}'.format(pop.lowest_fitness))
                print('Highest: {}'.format(pop.highest_fitness))
                print('Mean: {}\n'.format(pop.mean_fitness))
            
            self.io.get_data(i, pop)

            if self.elitism:
                best_ind = pop.best_ind
                new_pop.add_individuals([best_ind])

            while new_pop.size < pop.size:
                selected_ind = sel.select(pop)
                children = self.genetic_evolve(i, selected_ind)
                new_pop.add_individuals(children)
                if new_pop.size > pop.size:
                    new_pop.remove_last_individual()

            # UPDATE POPULATION
            pop = deepcopy(new_pop)