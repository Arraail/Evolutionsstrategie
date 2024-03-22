#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence produces only rudimentary progress messages and does not provide
batch distribution or timing prints, as `example_experiment2.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

>>> import cma  # doctest:+SKIP
>>> def fmin(fun, x0):
...     return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
import numpy as np
import numpy.random as rnd
import sys

def rank(x):
    return np.argsort(np.argsort(x))

def CommaSelection(parents,children):
    N = parents.shape[0]
    r = rank(children[:,-1])
    return children[r < N,:]

def TournamentSelection(parents,children,tournament_size=2):
    N = parents.shape[0]
    offspring = np.zeros(shape=parents.shape)
    for i in range(N):
        tournament = rnd.choice(children.shape[0], tournament_size)
        contestant1 = children[tournament[0]]
        contestant2 = children[tournament[1]]
        if contestant1[-1] < contestant2[-1]:
            offspring[i] = contestant1
        else:
            offspring[i] = contestant2
    return offspring

def RouletteWheel(parents,children):
    N = parents.shape[0]
    offspring = np.zeros(shape=parents.shape)
    fitness_sum = np.sum(children[:,-1])
    for i in range(N):
        r = rnd.uniform(0, fitness_sum)
        c = 0
        for x in children:
            c += x[-1]
            if c > r:
                offspring[i] = x
                break
    return offspring

def PlusSelection(parents,children):
    N = parents.shape[0]
    population = np.r_[parents,children]
    r = rank(population[:,-1])
    return population[r < N]

class IntermediateRecombination:
    def __init__(self, ρ):
        self._rho = ρ

    def __call__(self, population, λ):
        """Produce λ offspring from population by averaging the ρ best"""
        μ, D = population.shape
        children = np.zeros(shape=(λ, D))

        for l in range(λ):
            idx = rnd.choice(μ, replace=False, size=self._rho)
            children[l, :D] = np.mean(population[idx, :D], 0)
            children[l, -2] = np.median(population[idx, -2])
        # Clear function values
        children[:, -1] = np.nan
        return children

def BinaryRecombination(population, λ):
    """Produce λ offspring from population through binary recombination"""
    D = population.shape[1]
    children = np.zeros(shape=(λ, D))
    for l in range(λ):
        idx = rnd.choice(2, replace=False, size=2)
        sel = rnd.choice((idx), replace=True, size=D)
        children[l, :] = population[sel, np.arange(D)].copy()
    # Clear function values
    children[:, -1] = np.nan
    return children

class EvolutionaryStrategy:
    """Minimize a function w.r.t a continous parameter vectro `x` using an ES

    Parameters
    ----------
    μ: int
        Population size
    λ: int
        Number of offspring
    σ0: float
        Initial step size
    selection: callable
        Selection strategy used (PlusSelection or CommaSelection)
    recombination: callable
        Recombination operator used
    verbose: bool
        Output progress during optimization
    """

    def __init__(self, μ: int, λ: int, σ0: int,
                 selection=PlusSelection,
                 recombination=BinaryRecombination,
                 verbose=False):
        self._mu = μ
        self._lambda = λ
        self._sigma = σ0
        self._selection = selection
        self._recombination = recombination
        self._verbose = verbose

    def __call__(self, fn, x0, maxiter):
        """Minimize `fn` starting with initial solution `x0` for `maxiter` generations

        Parameters
        ----------
        fn: callable
            The objective function to be minimized.

                fun(x) -> float

            where x is a 1-D array with shape (n,).

        x0: ndarray, shape (n,)
            Initial guess. Array of real elements of size (n,), where n is the number of
            independent variables.

        maxiter: int
            Maximum number of iterations until the search terminates.

        Returns
        -------

        Dictionary with the following attributes:

        x: ndarray
            The solution of the optimization

        fun: float
            Value of objective function

        iter: int
            Generation in which the best function value was attained

        nfev: int
            Number of evaluations of the objective function `fn`

        nit: int
            Number of iterations (i.e. number of generations)
        """
        x0 = np.asarray(x0)
        D = len(x0)
        τ = 1 / np.sqrt(2 * D)

        # Each row of the matrix `population` encodes one individual. The first
        # D columns are the parameters, the next column the mutation strength /
        # step size and the last column the fitness value.
        #
        #   / x_1, x_2, x_3, ..., x_D, σ, y \  (1. individual)
        #   | x_1, x_2, x_3, ..., x_D, σ, y |  (2. individual)
        #   | ...                           |
        #   \ x_1, x_2, x_3, ..., x_D, σ, y /  (n. individual)
        #
        population = np.zeros(shape=(self._mu, D + 2))
        # Initial step size: σ = σ_0 * exp(τ * N(0, 1))
        population[:, D] = self._sigma * np.exp(τ * rnd.normal(0, 1, size=self._mu))
        # Initial population
        population[:, :D] = x0[np.newaxis, :]
        population[:, :D] += rnd.normal(0, population[:, -2][:, np.newaxis], size=(self._mu, D))
        population[:, -1] = np.apply_along_axis(fn, 1, population[:, :D])  # y = f(x)

        # Initial best solution
        min_idx = np.argmin(population[:, -1])
        best_x = population[min_idx, :D]
        best_fn = population[min_idx, -1]
        best_i = 0
        nfev = population.shape[0]
        nit = 0

        for i in range(maxiter):
            # Spawn a new generation of individuals
            offspring = self._recombination(population, self._lambda)

            # Mutate parameters and step size (self adaptation)
            offspring[:, :D] += np.random.normal(0, offspring[:, -2][:, np.newaxis], size=(self._lambda, D))
            offspring[:, -2] *= np.exp(τ * rnd.normal(0, 1, size=self._lambda))

            # Calculate fitness
            offspring[:, -1] = np.apply_along_axis(fn, 1, offspring[:, :D])

            # Update best ever seen solution
            min_idx = np.argmin(offspring[:, -1])
            if offspring[min_idx, -1] < best_fn:
                best_fn = offspring[min_idx, -1]
                best_x = offspring[min_idx, :D]
                best_i = i
                if self._verbose:
                    print(f"{i:3d}: fn={best_fn} x={best_x}")

            # Select new parents
            population = self._selection(population, offspring)

            # Update counters
            nfev += offspring.shape[0]
            nit += 1

        #return {"x": best_x, "fn": best_fn, "iter": best_i, "nfev": nfev, "nit": nit}
        return {"fn": best_fn, "iter": best_i, "nfev": nfev, "nit": nit}


### input
suite_name = "bbob"
output_folder = "scipy-optimize-fmin"
fmin = scipy.optimize.fmin
budget_multiplier = 1  # increase to 10, 100, ...

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
print(sys.argv[1])
### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    x0 = problem.initial_solution
    es = EvolutionaryStrategy(μ=10, λ=70, σ0=1,
                              selection=CommaSelection,
                              recombination=IntermediateRecombination(5))
    # apply restarts while neither the problem is solved nor the budget is exhausted
    while (problem.evaluations < problem.dimension * budget_multiplier
           and not problem.final_target_hit):
        #fmin(problem, x0, disp=False)  # here we assume that `fmin` evaluates the final/returned solution
        es(problem, x0, maxiter=50)
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

