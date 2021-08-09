import mbtk.utilities.experiment as util
import mbtk.utilities.functions as utilfunc

from mbtk.algorithms.mb.iamb import AlgorithmIAMB
from mbtk.algorithms.mb.ipcmb import AlgorithmIPCMB


class DsepAlgsEvalExpPathSet(util.ExperimentalPathSet):

    def __init__(self, root):
        super().__init__(root)
        self.BIFRepository = self.Root / 'bif_repository'


class DsepAlgsEvalExpSetup(util.ExperimentalSetup):

    def __init__(self):
        super().__init__()
        self.ExperimentDef = None
        self.ExDsDef = None
        self.Paths = None
        self.AlgorithmRunParameters = None
        self.Arguments = None

        algs = ['iamb', 'ipcmb']
        nets = ['alarm']
        self.AllowedBayesianNetworks = nets
        self.AllowedAlgorithms = algs


    def setup(self, arguments):
        self.Arguments = arguments

        self.AlgorithmName = self.Arguments.algorithm
        self.AlgorithmClass = get_algorithm_class(self.AlgorithmName)

        self.BayesianNetworkName = self.Arguments.bayesian_network
        bn_sourcepath = self.Paths.BIFRepository / self.BayesianNetworkName
        bn_sourcepath = bn_sourcepath.with_suffix('.bif')

        self.BayesianNetwork = utilfunc.read_bif_file(bn_sourcepath, use_cache=True)
        self.BayesianNetwork.finalize()


    def update_paths(self):
        super().update_paths()
        self.Paths.Datapoints = self.ExperimentDef.subfolder('algorithm_run_datapoints')
        self.Paths.CITestResultRepository = self.ExperimentDef.subfolder('ci_test_results')
        self.Paths.HResultRepository = self.ExperimentDef.subfolder('heuristic_results')
        self.Paths.Summaries = self.ExperimentDef.subfolder('summaries')
        self.Paths.Plots = self.ExperimentDef.subfolder('plots')


def get_algorithm_class(alg):
    if alg == 'iamb':
        return AlgorithmIAMB

    if alg == 'ipcmb':
        return AlgorithmIPCMB
