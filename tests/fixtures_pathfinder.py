import pytest
from pathlib import Path
import tests.utilities as testutil
import mbff.utilities.functions as util


@pytest.fixture(scope='session')
def bn_pathfinder():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'pathfinder.bif'), use_cache=True)
    bn.finalize()
    return bn



@pytest.fixture(scope='session')
def ds_pathfinder_4e3():
    """ The 'pathfinder' dataset with 4e3 samples"""
    configuration = dict()
    configuration['label'] = 'ds_pathfinder_4e3'
    configuration['sourcepath'] = testutil.bif_folder / 'pathfinder.bif'
    configuration['sample_count'] = int(4e3)
    configuration['random_seed'] = 97
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)