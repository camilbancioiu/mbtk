=====================
``BayesianNetwork``
=====================

Examples
--------

Simple example involving :class:`Variable` and :class:`ProbabilityDistribution`:

.. sourcecode:: python

  from mbff.dataset.BayesianNetwork import *

  # Variable AGE, with its simple probability distribution
  AGE = Variable('AGE')
  AGE.values = ['young', 'adult', 'old']
  AGE.probdist = ProbabilityDistribution(AGE)
  AGE.probdist.conditioning_variables = None    # Not a conditional probability distribution
  AGE.probdist.probabilities = {
    '<unconditioned>' : [0.3, 0.5, 0.2]         # Pr(young) = 0.3, Pr(adult) = 0.5, Pr(old) = 0.2
  }

  # Variable SEX, with its simple probability distribution
  SEX = Variable('SEX')
  SEX.values = ['M', 'F']
  SEX.probdist = ProbabilityDistribution(SEX)
  SEX.probdist.conditioning_variables = None    # Not a conditional probability distribution
  SEX.probdist.probabilities = {
    '<unconditioned>' : [0.5, 0.5]              # Pr(M) = 0.5, Pr(F) = 0.5
  }

  # Variable EDU, with probability distribution conditioned by AGE and SEX
  EDU = Variable('EDU')
  EDU.values = ['highschool', 'uni']
  EDU.probdist = ProbabilityDistribution(AGE)
  EDU.probdist.conditioning_variables = [AGE, SEX]
  EDU.probdist.probabilities = {
    ('young', 'M')    : [0.75, 0.25],           # e.g. Pr(uni | young, M) = 0.25
    ('young', 'F')    : [0.64, 0.36],           # e.g. Pr(uni | young, F) = 0.36
    ('adult', 'M')    : [0.72, 0.28],
    ('adult', 'F')    : [0.70, 0.30],
    ('old', 'M')      : [0.88, 0.12],
    ('old', 'F')      : [0.90, 0.10]
  }

  BN = BayesianNetwork('EDU-by-AGE-and-SEX')
  BN.variables = {
    'AGE' : AGE,
    'SEX' : SEX,
    'EDU' : EDU
  }

Module :mod:`mbff.dataset.BayesianNetwork`
------------------------------------------

.. autoclass:: mbff.dataset.BayesianNetwork.BayesianNetwork
  :members:

.. autoclass:: mbff.dataset.BayesianNetwork.Variable
  :members:

.. autoclass:: mbff.dataset.BayesianNetwork.ProbabilityDistribution
  :members:
