from argparse import ArgumentParser
from src.geneticSR import GeneticSR
import time
import random
import numpy as np
import json

EXEC_NUM = 30

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--results", required=True, type=str, help="Directory for results")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config_args = json.load(f)

    gsr = GeneticSR(config_args)
    for i in range(EXEC_NUM):
        print('EXECUTION {}\n'.format(i))
        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        gsr.run()

    results_path = 'results/' + args.results
    gsr.io.calculate_mean(EXEC_NUM)
    gsr.io.load_results(results_path)
