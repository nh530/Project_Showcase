# Kernelized Decision Trees and Random Forest
<img src="https://github.com/nh530/Project_Showcase/blob/master/Course_Work/Advanced_Machine_Learning/Research_Project/Images/forest.jpg" width="48">

## Author
* [Norman Hong](https://www.linkedin.com/in/norman-hong-b4075210a/)

## Table of contents
* [Introduction](#Introduction)
* [Setting Up Environment](#Setting-Up-Environment)
* [How to Run?](#How-to-Run?)
* [Code Style](#Code-Style)
* [References](#References)

## Introduction
Modern implementations of decision trees and random forests, like ID4 or CART, uses max voting or mean value functions in the leafs of each tree. This research project explores the potential of using kernel methods, similar to the implementation of kernel regressions, in the leaves. In essence, the idea is that instead of giving every training observation equal weight, kernels are used to give weight based on similarity. 

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

## Code style
Python Pep-8 standard 

## References
* [Salzberg, S. “A Nearest Hyperrectangle Learning Method.” Machine Learning, vol. 6, 251–276 (1991).
](https://doi.org/10.1023/A:1022661727670)
* [Ho, T.K. "Random decision forests." Proceedings of 3rd International Conference on Document Analysis and Recognition, vol. 1, 278-282 (1995). DOI: 10.1109/ICDAR.1995.598994](https://ieeexplore.ieee.org/document/598994)
* [Torgo, L. “Kernel Regression Trees,” Proceedings of the poster papers of the European Conference on Machine Learning, (1999).](https://www.researchgate.net/publication/2378537_Kernel_Regression_Trees)
* [Breiman, L. “Random Forests.” Machine Learning, vol. 45, 5–32 (2001).](https://doi.org/10.1023/A:1010933404324)
* [Olson, M. “Making Sense of Random Forest Probabilities: A Kernel Perspective.” arXiv, (2018).](https://arxiv.org/abs/1812.05792)
* [Olson, M. “Making Sense of Random Forest Probabilities: A Kernel Perspective.” arXiv, (2018).](https://arxiv.org/abs/1812.05792)
* [Denil, M., Freitas, N., Matheson, D. “Narrowing the Gap: Random Forests In Theory and In Practice.” arXiv, (2013).](https://arxiv.org/abs/1310.1415)
* [Scornet, E. “Random Forests and Kernel Methods.” arXiv, (2015).](https://arxiv.org/abs/1502.03836)
* [Geurts, P., Ernst, D. & Wehenkel, L. “Extremely randomized trees.” Machine Learning, 63, 3–42 (2006).](https://doi.org/10.1007/s10994-006-6226-1)
* [Itani, S., Lecron, F., Fortemps, P. “A One-Class Decision Tree Based on Kernel Density Estimation.” arXiv, (2020).](https://arxiv.org/abs/1805.05021)
* [Ahmad, A. “Decision Tree Ensembles based on Kernel Features.” Applied Intelligence, 41, 855-869 (2014).](https://doi.org/10.1007/s10489-014-0575-4)
* [Sarem. “A Brief Proof-Of-Concept for Kernelized Decision Trees.” Numbers and Code. Accessed 18 March 2020.](https://numbersandcode.com/a-brief-proof-of-concept-for-kernelized-decision-trees.)
* [Breiman, L. “Some infinity theory for predictor ensembles.” Technical Report, 577, UC Berkeley, (2000).](https://www.stat.berkeley.edu/~breiman/some_theory2000.pdf)

