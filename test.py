import unittest

from mbff_tests.TestDatasetMatrix import TestDatasetMatrix
from mbff_tests.TestRCV1v2DatasetSource import TestRCV1v2DatasetSource
from mbff_tests.TestBinarySyntheticDatasetSource import TestBinarySyntheticDatasetSource
from mbff_tests.TestBinaryExperimentalDataset import TestBinaryExperimentalDataset
from mbff_tests.TestExperimentalDataset import TestExperimentalDataset
from mbff_tests.TestInfoTheory import TestInfoTheory
from mbff_tests.TestAlgorithmIGt import TestAlgorithmIGt
from mbff_tests.TestAlgorithmRun import TestAlgorithmRun
from mbff_tests.TestExperimentRun import TestExperimentRun

if __name__ == '__main__':
    unittest.main(verbosity=2)

