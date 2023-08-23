import numpy as np
from itertools import cycle
from copy import deepcopy
import random

"""
An Individual has a genotype, which is represented as a list
of lists according to the following Structured Grammar 
(one list per "chromosome", where the value is the index 
for one of the possibilites), and it also has a tree representation.

<expr> : (<expr_lvl1><op><expr_lvl1>) 
<expr_lvl1> : (<expr_lvl2><op><expr_lvl2>) | var | const
<expr_lvl2> : (<expr_lvl3><op><expr_lvl3>)  | var | const
<expr_lvl3> : (<var><op><var>) | <preop>(<var>) | var | const
<op> : + | - | * | /
<preop> : sin | cos 
<var> : x | y
<const> : [-1, 1]
"""

# Grammar Definition
GRAMMAR = {
    'expr': [['(', 'expr_lvl1', 'op', 'expr_lvl1', ')']],
    'expr_lvl1': [['(', 'expr_lvl2', 'op', 'expr_lvl2', ')'],['var'], ['const']],
    'expr_lvl2': [['(', 'expr_lvl3', 'op', 'expr_lvl3', ')'], ['var'], ['const']],
    'expr_lvl3': [['(', 'var', 'op', 'var', ')'], ['preop', '(', 'var', ')'], ['var'], ['const']],
    'op': [['+'], ['-'], ['*'], ['/']],
    'preop': [['sin'], ['cos']],
    'var': [['x'], ['y']],
    'const': []
}

GENE_SIZE = {'expr': 1, 
             'expr_lvl1': 2, 
             'expr_lvl2': 4, 
             'expr_lvl3': 8,
             'op': 15,
             'preop': 8,
             'var': 16,
             'const': 8}

NON_TERMINALS = ['expr', 'expr_lvl1', 'expr_lvl2','expr_lvl3', 'op', 'preop', 'var', 'const']
RECURSIVE_EXPANSION = ['expr_lvl1', 'expr_lvl2','expr_lvl3']


class Node:
    def __init__(self, symbol):
        self.symbol = symbol 
        self.children = []
    
    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.symbol)+"\n"
        for child in self.children:
            ret += child.__repr__(level+1)
        return ret
    
    def get_str_expression(self):
        ret = ''
        if self.is_terminal():
            ret = str(self.symbol)
        for child in self.children:
            ret += child.get_str_expression()
        return ret
    
    def is_terminal(self):
        return self.symbol not in NON_TERMINALS


# Creates a random genotype. It has two types of initialization:
# grow - initializes completely random genes
# full - initializes only non-terminals until reaches max expansion
class Genotype:
    def __init__(self, type):
        self.genotype = {}
        self.iter = {}
        if type == 'grow':
            self.generate_grow_genotype()
        elif type == 'full':
            self.generate_full_genotype()
    
    def __repr__(self):
        return '{}'.format(self.genotype)

    def generate_grow_genotype(self):
        for gene in GRAMMAR:
            if gene == 'const':
                self.genotype[gene] = np.random.uniform(-1, 1, size=GENE_SIZE[gene])
                self.iter[gene] = cycle(self.genotype[gene])
            else: 
                self.genotype[gene] = np.random.randint(len(GRAMMAR[gene]), size=GENE_SIZE[gene])
                self.iter[gene] = cycle(self.genotype[gene])

    def generate_full_genotype(self):
        recursion = random.choice(RECURSIVE_EXPANSION)
        for gene in GRAMMAR:
            if gene == 'const':
                self.genotype[gene] = np.random.uniform(-1, 1, size=GENE_SIZE[gene])
                self.iter[gene] = cycle(self.genotype[gene])
            # Do not expand 'var' or 'const' nodes until reaches maximum expansion
            elif gene in RECURSIVE_EXPANSION and gene != recursion:
                self.genotype[gene] = np.random.randint(len(GRAMMAR[gene])-2, size=GENE_SIZE[gene])
                self.iter[gene] = cycle(self.genotype[gene])
            else: 
                self.genotype[gene] = np.random.randint(len(GRAMMAR[gene]), size=GENE_SIZE[gene])
                self.iter[gene] = cycle(self.genotype[gene])

    def get_expansion(self, symbol):
        if symbol == 'const':
            return [next(self.iter[symbol])]
        if symbol in NON_TERMINALS:
            return GRAMMAR[symbol][next(self.iter[symbol])]
        return []

    def generate_tree(self, node=None):
        if not node:
            node = Node('expr')
        expansion = self.get_expansion(node.symbol)
        for symbol in expansion:
            child = Node(symbol)
            node.children.append(child)
            self.generate_tree(child)
        return node
    
    def crossover(self, partner):
        mask = {gene: np.random.randint(2) for gene in GRAMMAR}
        offspring1 = deepcopy(self)
        offspring2 = deepcopy(partner)
        for gene in mask:
            if mask[gene] == 1:
                offspring1.genotype[gene] = partner.genotype[gene].copy()
                offspring1.iter[gene] = cycle(offspring1.genotype[gene])
                offspring2.genotype[gene] = self.genotype[gene].copy()
                offspring2.iter[gene] = cycle(offspring2.genotype[gene])
        return [offspring1, offspring2]

    def mutation(self):
        offspring = deepcopy(self)
        target_gene = random.choice(list(GRAMMAR.keys()))
        target_idx = np.random.randint(len(offspring.genotype[target_gene]))
        if target_gene == 'const':
            offspring.genotype[target_gene][target_idx] = np.random.uniform(-1, 1)
        else:
            offspring.genotype[target_gene][target_idx] = np.random.randint(len(GRAMMAR[target_gene]))
        return offspring