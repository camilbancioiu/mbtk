# MBFF
This is an experimental framework for studying **M**arkov **B**lanket **F**eature **F**ilters. It contains an original implementation of an old, yet canonical feature selection algorithm, based on Markov blankets: Koller and Sahami's algorithm (KS). The trivial Information Gain Thresholding algorithm is implemented as well, as a baseline, even if it doesn't use Markov blankets.

This implementation provides a few efficiency optimizations of the KS algorithm. They can be enabled or disabled at will. The framework also defines a specific set of experiments to evaluate the efficiency gains of these optimizations.

*Please submit any comments, improvement suggestions and bugs as Issues of this repository.*

## Documentation
Will follow soon.

## Requirements
1. The framework is written in Python 3. Make sure you have it installed. 
1. The Python 3 module `sklearn` is required. Please install it according to the instructions here: https://scikit-learn.org/stable/install.html.
1. The LYRL2004 version of the Reuters Corpus Volume 1 is required, which will be downloaded by the `prepare` script, bundled with the framework. Please run `./prepare` before attempting anything else. A download of approximately 150 MB will start.

