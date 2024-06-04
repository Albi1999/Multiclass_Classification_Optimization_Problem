# Multiclass Logistic Regression Optimization Problem

This repository contains my university project of the course Optimization For Data Science
The project is a collaborative effort by Alberto Calabrese, Greta d'Amore Grelli, Eleonora Mesaglio, and Marlon Helbing.

## Project Overview

**Optimization Problem**: The core of this project is to solve the multi-class classification problem of the form:
$$
\min_{X\in\R^{d\times k}} f(x) = \min_{X\in\R^{d\times k}} \sum_{i=1}^{m}\left[-x_{b_i}^Ta_i + \log\left(\sum_{c=1}^{k}\exp(x_c^Ta_i)\right)\right],
$$

where $a_i\in\R^d$ are the features of the $i$-th sample, $x_c\in\R^d$ is the column vector of the matrix of parameters $X\in\R^{d\times k}$ relating to class $c$ and $b_i\in\{1,\dots k\}$ is the label associated to the $i$-th sample, given by the following probability:

$$
P(b_i | X, a_i) = \frac{\exp(x_{b_i}^Ta_i)}{\sum_{c=1}^{k}\exp(x_c^Ta_i)}.
$$

**Gradient Descent (GD)**: One of the primary methods used in this project for solving the optimization problem is Gradient Descent. This iterative optimization algorithm is used to find the minimum of the objective function.

**Block Coordinate Descent (BGCD) - Randomized and Gauss-Southwell**: The project also employs Block Coordinate Descent methods, both in its standard form and a variant using Gauss-Southwell rule. These methods involve updating one coordinate at a time, which can be computationally efficient for certain problems.

## Contributing

This project is a part of a university assignment and is not open for contributions.
