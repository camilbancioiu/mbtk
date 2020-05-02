import mbff_tests.utilities as testutil
import pytest
from pathlib import Path
import mbff.utilities.functions as util

testfolder = Path('mbff_tests', 'testfiles')
bif_folder = Path('mbff_tests', 'bif_files')


@pytest.fixture
def ds_survey_small():
    """The 'survey' dataset with 2e4 samples"""
    configuration = dict()
    configuration['label'] = 'survey'
    configuration['sourcepath'] = Path(bif_folder, 'survey.bif')
    configuration['sample_count'] = int(2e4)
    configuration['random_seed'] = 42 * 42
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['folder'] = Path(testfolder, 'tmp', 'mockdatasets')

    return testutil.make_test_dataset(configuration)



@pytest.fixture
def bn_alarm():
    bn = util.read_bif_file(Path(bif_folder, 'alarm.bif'))
    bn.finalize()
    return bn
