<h1 align="center"> Genetic Symbolic Regression </h1>

The challenge posed by Symbolic Regression revolves around discovering a mathematical expression that optimally fits a collection of samples. These samples encompass values from $n$ input variables alongside an associated output value. This pursuit involves a fusion of fundamental operations (addition, subtraction, multiplication, and division), trigonometric functions, logarithms, and more. In essence, Symbolic Regression extends and refines the notions of linear or polynomial regression.

One of the approaches commonly found in literature for seeking an approximate solution for this problem involves employing genetic programming, which is implemented here. This methodology is executed by establishing an initial population of individuals, where each individual represents a potential solution to the problem (often presented as trees, depicting functions). Subsequently, all individuals undergo fitness evaluation, a subset is selected, and genetic operators (crossover and mutation) are applied among them, resulting in a new population. This process continues until a maximum number of generations has been generated or some other stopping criterion has been met.

## Execution

The program can be executed using the following command line:

```bash
python main.py --config path_to_config_file --results path_to_results_dir
```

## Config File

An essential step for running the algorithm is to create a configuration file in JSON format, which includes the population size, number of generations, paths for test and training data, selection method, probabilities of genetic operators and indicators for utilizing elitism and logging. An example of a possible config file is shown below:

| :exclamation:  The current version of the program only accepts training data with 2 variables. Therefore, "var_num" must be set to 2.  |
|-----------------------------------------|

```json
{
    "var_num": 2,
    "population_size": 250,
    "generations": 250,
    "train_file": "datasets/synth1/synth1-train.csv",
    "test_file": "datasets/synth1/synth1-test.csv",
    "selection":{
        "name": "roulette",
        "args": {
            "k": 2
        }
    },
    "mutation_prob": 0.20,
    "crossover_prob": 0.75,
    "elitism": true,
    "verbose": true
}
```
