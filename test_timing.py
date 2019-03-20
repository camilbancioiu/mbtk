import random
import timeit
from pathlib import Path
from collections import Counter

import numpy
import unittest

from mbff_tests.TestBase import TestBase

import mbff.utilities.functions as util
from mbff.dataset.BayesianNetwork import *

TESTCOUNT = 40000

print('Timing test starts...')
print()


survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
bn = util.read_bif_file(survey_bif)
bn.finalize()

timer = timeit.Timer('bn.sample()', globals=globals())
print('Time without optimal sampling order:...')
time_reference = timer.timeit(TESTCOUNT)
print(time_reference)


print()


survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
bn = util.read_bif_file(survey_bif)
bn.variable_names__sampling_order = ['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN']
bn.finalize()

timer = timeit.Timer('bn.sample()', globals=globals())
print('Time with optimal sampling order:...')
time = timer.timeit(TESTCOUNT)
print(time, '({:.3f}% of reference time)'.format(time * 100 / time_reference))


print()


survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
bn = util.read_bif_file(survey_bif)
bn.variable_names__sampling_order = ['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN']
bn.finalize()

timer = timeit.Timer('bn.sample(values_as_indices=True)', globals=globals())
print('Time with optimal sampling order, as indices:...')
time = timer.timeit(TESTCOUNT)
print(time, '({:.3f}% of reference time)'.format(time * 100 / time_reference))
