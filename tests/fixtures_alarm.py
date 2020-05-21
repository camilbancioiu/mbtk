import pytest
from pathlib import Path
import tests.utilities as testutil
from mbff.structures.ADTree import ADTree
import mbff.utilities.functions as util


@pytest.fixture(scope='session')
def bn_alarm():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'alarm.bif'))
    bn.finalize()
    return bn



@pytest.fixture(scope='session')
def ds_alarm_4e4():
    """ The 'alarm' dataset with 4e4 samples"""
    configuration = dict()
    configuration['label'] = 'ds_alarm_4e4'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(4e4)
    configuration['random_seed'] = 235711131719
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_alarm_8e3():
    """ The 'alarm' dataset with 8e3 samples"""
    configuration = dict()
    configuration['label'] = 'ds_alarm_8e3'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(8e3)
    configuration['random_seed'] = 97
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_alarm_3e3():
    configuration = dict()
    configuration['label'] = 'ds_alarm_3e3'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(3e3)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['random_seed'] = 1984 * 1984
    configuration['method'] = 'random'
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_alarm_5e2():
    configuration = dict()
    configuration['label'] = 'ds_alarm_5e2'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(5e2)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['random_seed'] = 1985
    configuration['method'] = 'random'
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_alarm_3e2():
    configuration = dict()
    configuration['label'] = 'ds_alarm_3e2'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(3e2)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['random_seed'] = 1983
    configuration['method'] = 'random'
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def adtree_alarm_8e3_llta0(ds_alarm_8e3):
    configuration = dict()
    configuration['label'] = 'adtree_alarm_8e3_llta0'
    configuration['ci_test_ad_tree_class'] = ADTree
    configuration['leaf_list_threshold'] = 0
    return testutil.prepare_AD_tree(configuration, ds_alarm_8e3.datasetmatrix)



@pytest.fixture(scope='session')
def adtree_alarm_3e3_llta100(ds_alarm_3e3):
    configuration = dict()
    configuration['label'] = 'adtree_alarm_3e3_llta100'
    configuration['ci_test_ad_tree_class'] = ADTree
    configuration['leaf_list_threshold'] = 100
    return testutil.prepare_AD_tree(configuration, ds_alarm_3e3.datasetmatrix)



@pytest.fixture(scope='session')
def adtree_alarm_5e2_llta0(ds_alarm_5e2):
    configuration = dict()
    configuration['label'] = 'adtree_alarm_5e2_llta20'
    configuration['ci_test_ad_tree_class'] = ADTree
    configuration['leaf_list_threshold'] = 20
    return testutil.prepare_AD_tree(configuration, ds_alarm_5e2.datasetmatrix)



@pytest.fixture(scope='session')
def adtree_alarm_3e2_llta0(ds_alarm_3e2):
    configuration = dict()
    configuration['label'] = 'adtree_alarm_3e2_llta0'
    configuration['ci_test_ad_tree_class'] = ADTree
    configuration['leaf_list_threshold'] = 0
    return testutil.prepare_AD_tree(configuration, ds_alarm_3e2.datasetmatrix)
