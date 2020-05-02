import mbff_tests.utilities as testutil
import pytest
from pathlib import Path

testfolder = Path('mbff_tests', 'testfiles')


@pytest.fixture
def ds_survey_small():
    """The 'survey' dataset with 2e4 samples"""
    configuration = dict()
    configuration['label'] = 'survey'
    configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
    configuration['sample_count'] = int(2e4)
    configuration['random_seed'] = 42 * 42
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['folder'] = Path(testfolder, 'tmp', 'mockdatasets')

    return testutil.make_test_dataset(configuration)
