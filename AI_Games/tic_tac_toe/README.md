# Machine Learning for Tic-Tac-Toe
<p align='center'>
<img src="https://github.com/nh530/Project_Showcase/blob/master/AI_Games/tic_tac_toe/images/ttt.png" width="500">
</p>

## Author

* [Norman Hong](https://www.linkedin.com/in/norman-hong-b4075210a/)

## Table of contents

* [Introduction](#Introduction)
* [Results](#Results)
* [Setting Up Environment](#Setting-Up-Environment)
* [How to Run?](#How-to-Run?)
* [TODO](#To-do)

## Introduction

* Tic-Tac-Toe is an old game with a known solution, minmax algorithm with alpha-beta pruning. The goal of this project is to explore the use of machine learning to create an agent that is capable of learning the optimal solution. 

* tic_tac_toe_emulator.py contains the script necessary to run a Tic-Tac-Toe game. 

* linear_regression_ai.py is a machine learning implementation that decides on the next optimal move, given the current Tic-Tac-Toe board, by using linear regression. A set of features is derived from the board, and the target variable is a number ranging from -100 to 100 where 100 is the value if win, and -100 is the value if lost. 

## Results

* Linear regression does not perform better than alpha-beta pruning. It can work somewhat well depending on the board features used, but at best, it only learns to draw the game.

## Setting Up Environment

1. Installing virtual environment package 
``` bash
python3 -m pip install --user virtualenv
```

2. Create a Python virtual environment called "anly601_finalproject" using `conda`
``` bash
$ python3 -m venv myEnv
```

3. Activate virtual environment
``` bash
./env/Scripts/activate
```

4. Clone this folder and install required packages
There is no dependencies with other files in this repo.
``` bash
$ python3 -m pip install -r requirements.txt
```

## How to Run?

``` bash
$ python3 Decision_Tree_Kernel_leaves.py
```

## To do
1. Logistic Regresssion
2. Boosting tree
3. Reinforcement learning
