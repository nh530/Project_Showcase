# Kernelized Decision Trees and Random Forest
<p align='center'>
<img src="https://github.com/nh530/Project_Showcase/blob/master/Research_Project/Images/forest.jpg" width="500">
</p>

## Author

* [Norman Hong](https://www.linkedin.com/in/norman-hong-b4075210a/)
* This work was part of a larger project in partnership with [Jack Hart](https://www.linkedin.com/in/jack-hrt/)

## Table of contents

* [Introduction](#Introduction)
* [Setting Up Environment](#Setting-Up-Environment)
* [Related Works](#Related-Works)
* [Method](#Method-or-Algorithm)
* [How to Run?](#How-to-Run?)
* [Code Style](#Code-Style)
* [References](#References)

## Introduction

Modern implementations of decision trees and random forests, like ID4 or CART, uses max voting or mean value functions in the leafs of each tree. This research project explores the potential of using kernel methods, similar to the implementation of kernel regressions, in the leaves. In essence, the idea is that instead of giving every training observation equal weight, kernels are used to give weight based on similarity. Decision trees are methods that have low bias and overfit to the training data. As such, a way to reduce the bias is to use kernel weights at the leaves, which acts as a smoothing mechanism. Therefore, reducing the bias and giving the user another tool to control the bias-variance tradeoff that is common among machine learning algorithms.

## Related Works

Kernel regression trees have been defined by \cite{Torgo_1999} as a decision tree model that applies kernel regressions in tree leaves (as opposed to just saving the class averages). The leaf nodes of a decision tree can be viewed as segmented hyperrectangles of the feature space \cite{Salzberg_1991}. Therefore combining decision trees with kernels lowers the computational cost of classification, compared to a regular kernel regression, because only the data points in a given node are used in the calculation. The issue with kernel methods is that it requires storing all training instances and sacrifices scalability. In this age of big data, this would pose a big issue. Torgo acknowledged that future work should attempt techniques to generalize and compress data in the leaves for this reason. The focus of this project is on continuing Torgo's work and presenting a novel way to compress the data. 

## Method or Algorithm

Modern implementations of random forests and decision trees use max voting in leaves for classification tasks and averaging in  leaves for regression tasks. When the leaves are impure, there could be more information to be gained in each leaf. Kernel regressions are a popular non-parametric technique for estimating non-linear relationships between random variables. The proposed method, called kernel decision trees, integrates decision trees with kernel weighted averaging at the leaves such that you don’t have to save all n training points. 

<p align='center'>
<img src="https://github.com/nh530/Project_Showcase/blob/master/Research_Project/Images/dec_tree_kernels.png" width="500">
</p>

Terminal nodes of a decision tree can be interpreted as hyperrectangle subspaces of the feature space where any two points belonging to the same hyperrectangle are given the same value. A way to generalize the kernel computation is to compute the distance of a point to the edges. This is in contrast to computing all the distances of a given point to all the other points in a given subspace. As a result, only the edges of a hyperrectangle, which are also the decision boundaries in a tree, need to be saved, and the resulting prediction is some kernelized weighted average. 

An issue with this method is storing decision boundaries and then determine which points on the boundary to use during the kernel computation. Another issue is that the results of the smoothing will be strongly correlated with the structure of the decision tree. Intuitively, this would minimize the effects of using a weighted smoothing. From this point of view, consider saving a subset of the training data by selecting the points that are closest to the boundary edges. Given a training dataset with vectors of d dimensions, a boundary point is defined to be any point where the values in one of its vector components is c away from a cutoff value in the same dimension. Therefore, the proposed method will first create a decision tree based on the CART algorithm. Secondly, it will iterate through all training instances 1 time, run the instance vector through the generated decision tree, calculate if the vector component is at least c away from the cutoff value at each node. Lastly, all boundary points are saved and used for kernel computation during predictions. 

## Conclusion

The goal of this work was to fix some of the limitations of decision trees in the hopes of introducing an algorithm that is suitable for large datasets, while also not sacrificing performance. The presented algorithm was tested on small real world datasets
that demonstrated a clear advantage. Kernel decision trees outperform a decision tree across all of the tested datasets, although only slightly, and was hypothesized that it was due to the simple and small nature of these datasets. As a result, this is a clear improvement and expands upon the work done by Torgo in the Kernel Regression Trees paper. 

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
> All links working as of April 6, 2020
