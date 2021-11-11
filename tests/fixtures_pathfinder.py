import pytest
from pathlib import Path
import tests.utilities as testutil
from mbtk.structures.BayesianNetwork import BayesianNetwork


@pytest.fixture(scope='session')
def bn_pathfinder():
    path = Path(testutil.bif_folder, 'pathfinder.bif')
    bn = BayesianNetwork.from_bif_file(path, use_cache=True)
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
