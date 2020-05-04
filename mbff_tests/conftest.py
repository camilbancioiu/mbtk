import pytest
from pathlib import Path
import mbff_tests.utilities as testutil
import mbff.utilities.functions as util


@pytest.fixture
def ds_survey_small():
    """The 'survey' dataset with 2e4 samples"""
    configuration = dict()
    configuration['label'] = 'survey'
    configuration['sourcepath'] = Path(testutil.bif_folder, 'survey.bif')
    configuration['sample_count'] = int(2e4)
    configuration['random_seed'] = 42 * 42
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['folder'] = Path(testutil.tmp_folder, 'mockdatasets')

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
