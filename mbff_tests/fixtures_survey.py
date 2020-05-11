import pytest
from pathlib import Path
import mbff_tests.utilities as testutil
import mbff.utilities.functions as util


@pytest.fixture(scope='session')
def bn_survey():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'survey.bif'))
    bn.finalize()
    return bn



@pytest.fixture(scope='session')
def ds_survey_2e4():
    """The 'survey' dataset with 2e4 samples"""
    configuration = dict()
    configuration['label'] = 'ds_survey_2e4'
    configuration['sourcepath'] = testutil.bif_folder / 'survey.bif'
    configuration['sample_count'] = int(2e4)
    configuration['random_seed'] = 42 * 42
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_survey_2e3():
    """The 'survey' dataset with 2e3 samples"""
    configuration = dict()
    configuration['label'] = 'ds_survey_2e3'
    configuration['sourcepath'] = testutil.bif_folder / 'survey.bif'
    configuration['sample_count'] = int(2e3)
    configuration['random_seed'] = 42 * 42 + 1
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_survey_5e2():
    """The 'survey' dataset with 5e2 samples"""
    configuration = dict()
    configuration['label'] = 'ds_survey_5e2'
    configuration['sourcepath'] = testutil.bif_folder / 'survey.bif'
    configuration['sample_count'] = int(5e2)
    configuration['random_seed'] = 44 * 40 + 1
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def adtree_survey_2e3_llta0(ds_survey_2e3):
    configuration = dict()
    configuration['label'] = 'adtree_survey_2e3_llta0'
    configuration['leaf_list_threshold'] = 0
    configuration['debug'] = 1
    return testutil.prepare_AD_tree(configuration, ds_survey_2e3.datasetmatrix)



@pytest.fixture(scope='session')
def adtree_survey_5e2_llta20(ds_survey_5e2):
    configuration = dict()
    configuration['label'] = 'adtree_survey_5e2_llta20'
    configuration['leaf_list_threshold'] = 20
    configuration['debug'] = 1
    return testutil.prepare_AD_tree(configuration, ds_survey_5e2.datasetmatrix)
