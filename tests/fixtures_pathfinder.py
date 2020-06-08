import pytest
from pathlib import Path
import tests.utilities as testutil
import mbff.utilities.functions as util


@pytest.fixture(scope='session')
def bn_pathfinder():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'pathfinder.bif'), use_cache=True)
    bn.finalize()
    return bn
