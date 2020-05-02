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
        TestBase.TestsTagsToExclude.append('ipcmb_run')
        del sys.argv[1]
    elif test_profile == 'all':
        del sys.argv[1]


    # After deciding what tests to exclude, we can now import them. Importing
    # AFTER setting TestsTagsToExclude is required, because unittest.skipIf
    # decorators are evaluated at import time.

    from mbff_tests.test_DatasetMatrix import TestDatasetMatrix
    from mbff_tests.test_RCV1v2DatasetSource import TestRCV1v2DatasetSource
    from mbff_tests.test_BinarySyntheticDatasetSource import TestBinarySyntheticDatasetSource
    from mbff_tests.test_BinaryExperimentalDataset import TestBinaryExperimentalDataset
    from mbff_tests.test_ModelBuildingExperimentalDataset import TestModelBuildingExperimentalDataset
    from mbff_tests.test_InfoTheory import TestInfoTheory
    from mbff_tests.test_AlgorithmIGt import TestAlgorithmIGt
    from mbff_tests.test_AlgorithmRun import TestAlgorithmRun
    from mbff_tests.test_ExperimentRun import TestExperimentRun
    from mbff_tests.test_BIFReading import TestBIFReading
    from mbff_tests.test_BayesianNetwork import TestBayesianNetwork
    from mbff_tests.test_SampledBayesianNetworkDatasetSource import TestSampledBayesianNetworkDatasetSource
    from mbff_tests.test_VariableAndPMF import TestVariableAndPMF
    from mbff_tests.test_GTestUnoptimized import TestGTestUnoptimized
    from mbff_tests.test_ADTree import TestADTree
    from mbff_tests.test_AlgorithmIPCMB import TestAlgorithmIPCMB
    from mbff_tests.test_AlgorithmIPCMBWithGtests import TestAlgorithmIPCMBWithGtests

    import unittest
    unittest.main(verbosity=2)
