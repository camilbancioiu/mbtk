# The Markov Boundary Toolkit (MBTK)

MBTK is a library that assists in the research and development of Markov boundary discovery algorithms. It contains helpful tools, such as a Bayesian network sampler, data set generators and various components used to configure and run various experiments.

The MBTK itself was already used to build successful experiments. Most notably, the `experiments/experiment_dcmi` folder in MBTK contains the configuration and customizations needed to evaluate the `dcMI` optimization for the computation of the G-test, as part of a comparative experiment involving the IPC-MB algorithm (article currently in review). A previous iteration of MBTK (see [MBFF](https://github.com/camilbancioiu/MBFF)) was used to build an experiment that evaluated 4 individual optimizations for the older KS algorithm ([published at ROMJIST](https://www.romjist.ro/abstract-620.html), an open-access journal).

For a demonstration of MBTK, see [below](#demonstration).

## Dependencies

The following Python packages are imported by MBTK. 
* `numpy`
* `scipy`
* `pytest`
* `lark-parser`
* `pympler`
* `matplotlib`
* `pudb`

Please install them with `pip install -r requirements.txt`. 

Alternatively, use `pip install numpy scipy pytest lark-parser pympler matplotlib pudb` directly.

## Components

The MBTK contains many utilities, but also complete implementations of algorithms, data structures and optimizations.

### Utilities

* Bayesian Interchange Format (BIF) reader, which instantiates a Bayesian network from a BIF file.
* Bayesian network sampler, which generates large amounts of fully defined samples from a Bayesian network, i.e. each sample contains a value for each variable in the network.
* Data set generators with multiple sources: Bayesian networks, synthetic binary variables or the [Reuters Corpus Volume 1 v2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm).
* Experiment infrastructure utilities, such as experimental definition classes and algorithm runners, which manage and measure individual algorithm runs and also collect their results.

### Data structures

* Bayesian network, which can be constructed either from a BIF file or programmatically; includes an implementation of the d-separation algorithm to test for conditional independence.
* AD-trees, which can represent an entire data set completely and allow for very fast access to sample counts, but can consume a lot of memory; both static and dynamic AD-trees are provided (see the [static AD-tree article](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.72.4560) and the [dynamic AD-tree article](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.64)); data structures used by the AD-trees are also provided, namely implementations for contingency tables and contingency trees.

### Algorithms

* Iterative Parent-Child Markov Blanket algorithm (IPC-MB), implemented in a simple and general way, isolated from any independence test; for details on IPC-MB, see the [original article](https://link.springer.com/chapter/10.1007%2F978-3-540-68825-9_10).

### Mathematics

* G-test implemented in the canonical form; performs queries on the data set alone; used mostly as a baseline for efficiency comparisons.
* G-test implemented with AD-trees (configurable for static or dynamic tree); when using the static AD-tree, the data set is not queried for independence tests at all, but the tree must be built completely before the algorithm starts; when using the dynamic AD-tree, no building step is required before the algorithm starts, but access to the data set becomes required when a query cannot be satisfied by the tree alone.
* G-test implemented with the `dcMI` optimization, which decomposes the `G` statistic into joint entropy terms and caches them for future reuse; access to the data set is required when a test cannot be computed with the cached terms alone.

## Demonstration

A demonstrative experiment can be performed by running `python demo.py` in the root MBTK folder. The `demo.py` script will run the IPC-MB algorithm for each of the 37 variables of a synthetic data set. This experiment consists of two runs: one where IPC-MB is optimized with a dynamic AD-tree and one where IPC-MB is optimized with `dcMI`. This demonstrative experiment compares `dcMI` and the dynamic AD-tree, highlighting their difference in efficiency when added to IPC-MB.
