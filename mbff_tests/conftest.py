import pytest
from pathlib import Path
import mbff_tests.utilities as testutil
import mbff.utilities.functions as util


@pytest.fixture
def ds_survey_2e4():
    """The 'survey' dataset with 2e4 samples"""
    configuration = dict()
    configuration['label'] = 'ds_survey_2e4'
    configuration['sourcepath'] = testutil.bif_folder / 'survey.bif'
    configuration['sample_count'] = int(2e4)
    configuration['random_seed'] = 42 * 42
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['folder'] = testutil.tmp_folder / 'mockdatasets'
    return testutil.make_test_dataset(configuration)



@pytest.fixture
def ds_lungcancer_4e4():
    """ The 'lungcancer' dataset with 4e4 samples"""
    configuration = dict()
    configuration['label'] = 'ds_lungcancer_4e4'
    configuration['sourcepath'] = testutil.bif_folder / 'lungcancer.bif'
    configuration['sample_count'] = int(4e4)
    configuration['random_seed'] = 129
    configuration['values_as_indices'] = False
    configuration['objectives'] = []
    configuration['folder'] = testutil.tmp_folder / 'mockdatasets'
    return testutil.make_test_dataset(configuration)



@pytest.fixture
def ds_lc_repaired_4e4():
    configuration = dict()
    configuration['label'] = 'ds_lc_repaired_4e4'
    configuration['sourcepath'] = testutil.bif_folder / 'lc_repaired.bif'
    configuration['sample_count'] = int(4e4)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['method'] = 'exact'
    configuration['random_seed'] = 1984
    return testutil.make_test_dataset(configuration)



@pytest.fixture
def ds_alarm_8e3():
    """ The 'alarm' dataset with 8e3 samples"""
    configuration = dict()
    configuration['label'] = 'ds_alarm_8e3'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(8e3)
    configuration['random_seed'] = 129
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['folder'] = testutil.tmp_folder / 'mockdatasets'
    return testutil.make_test_dataset(configuration)



@pytest.fixture
def ds_alarm_3e3():
    configuration = dict()
    configuration['label'] = 'ds_alarm_3e3'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(3e3)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['random_seed'] = 1984
    configuration['method'] = 'random'
    return testutil.make_test_dataset(configuration)



@pytest.fixture
def ds_alarm_3e2():
    configuration = dict()
    configuration['label'] = 'ds_alarm_3e2'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(3e2)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['random_seed'] = 1983
    configuration['method'] = 'random'
    return testutil.make_test_dataset(configuration)



@pytest.fixture
def bn_alarm():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'alarm.bif'))
    bn.finalize()
    return bn



@pytest.fixture
def bn_survey():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'survey.bif'))
    bn.finalize()
    return bn



@pytest.fixture
def bn_lungcancer():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'lungcancer.bif'))
    bn.finalize()
    return bn



@pytest.fixture
def bn_lc_repaired():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'lc_repaired.bif'))
    bn.finalize()
    return bn



@pytest.fixture
def adtree_alarm_3e3_llta100():
    configuration = dict()
    configuration['label'] = 'adtree_alarm_3e3_llta100'
    configuration['leaf_list_threshold'] = 100
    configuration['debug'] = 1
    return prepare_AD_tree(configuration)
