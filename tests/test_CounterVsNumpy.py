import math
import numpy
from collections import Counter

def test_counting_efficiency__Counter(ds_alarm_8e3):
    matrix = ds_alarm_8e3.datasetmatrix
    variables = matrix.get_variables('X', [2, 4, 6, 8])

    instances = variables.instances()
    counts = Counter(instances)
    print(counts)
