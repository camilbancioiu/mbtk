import sys

if __name__ == '__main__':
    from mbff_tests.TestBase import TestBase
    TestBase.TestsTagsToExclude = []

    try:
        test_profile = sys.argv[1]
    except IndexError:
        test_profile = 'all'

    print('Running test profile \'{}\''.format(test_profile))

    if test_profile == 'quick':
        TestBase.TestsTagsToExclude.append('sampling')
        TestBase.TestsTagsToExclude.append('conditional_independence')
        del sys.argv[1]
    elif test_profile == 'all':
        del sys.argv[1]


    # After deciding what tests to exclude, we can now import them. Importing
    # AFTER setting TestsTagsToExclude is required, because unittest.skipIf
    # decorators are evaluated at import time.

    from mbff_tests.TestDatasetMatrix import TestDatasetMatrix
    from mbff_tests.TestRCV1v2DatasetSource import TestRCV1v2DatasetSource
    from mbff_tests.TestBinarySyntheticDatasetSource import TestBinarySyntheticDatasetSource
    from mbff_tests.TestBinaryExperimentalDataset import TestBinaryExperimentalDataset
    from mbff_tests.TestModelBuildingExperimentalDataset import TestModelBuildingExperimentalDataset
    from mbff_tests.TestInfoTheory import TestInfoTheory
    from mbff_tests.TestAlgorithmIGt import TestAlgorithmIGt
    from mbff_tests.TestAlgorithmRun import TestAlgorithmRun
    from mbff_tests.TestExperimentRun import TestExperimentRun
    from mbff_tests.TestBIFReading import TestBIFReading
    from mbff_tests.TestBayesianNetwork import TestBayesianNetwork
    from mbff_tests.TestSampledBayesianNetworkDatasetSource import TestSampledBayesianNetworkDatasetSource
    from mbff_tests.TestVariableAndPMF import TestVariableAndPMF
    from mbff_tests.TestGStat import TestGStat
    from mbff_tests.TestADTree import TestADTree
    from mbff_tests.TestAlgorithmIPCMB import TestAlgorithmIPCMB

    import unittest
    unittest.main(verbosity=2)

