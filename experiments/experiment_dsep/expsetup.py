import re

import mbtk.utilities.experiment as util
from mbtk.structures.BayesianNetwork import BayesianNetwork

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
        self.AlgorithmName = ""
        self.AlgorithmClass = None
        self.BayesianNetworkName = ""
        self.BayesianNetwork = None
        self.Source = ""
        self.SampleCount = None
        self.SampleCountString = ""

        self.Sources = ['bn', 'ds']
        self.BayesianNetworks = ['alarm', 'mildew', 'child']
        self.Algorithms = ['iamb', 'ipcmb']


    def setup(self, arguments):
        self.Arguments = arguments

        self.AlgorithmName = self.Arguments.algorithm
        self.AlgorithmClass = get_algorithm_class(self.AlgorithmName)

        self.Source = self.Arguments.source_type

        self.BayesianNetworkName = self.Arguments.source_name
        bn_sourcepath = self.Paths.BIFRepository / self.BayesianNetworkName
        bn_sourcepath = bn_sourcepath.with_suffix('.bif')

        self.BayesianNetwork = BayesianNetwork.from_bif_file(bn_sourcepath, use_cache=True)
        self.BayesianNetwork.finalize()

        if self.Source == 'ds':
            if self.Arguments.sample_count is None:
                raise ValueError('sample count required')
            self.SampleCountString = self.Arguments.sample_count
            self.SampleCount = int(float(self.SampleCountString))


    def update_paths(self):
        super().update_paths()
        self.Paths.Datapoints = self.ExperimentDef.subfolder('algorithm_run_datapoints')
        self.Paths.CITestResultRepository = self.ExperimentDef.subfolder('ci_test_results')
        self.Paths.HResultRepository = self.ExperimentDef.subfolder('heuristic_results')
        self.Paths.Summaries = self.ExperimentDef.subfolder('summaries')
        self.Paths.Plots = self.ExperimentDef.subfolder('plots')


    def validate_arguments(self, arguments):
        self.validate_source(arguments.source)
        self.validate_source_name(arguments.source)
        self.validate_sample_count_string(arguments.sample_count)


    def validate_source_name(self, source_name):
        if source_name not in self.BayesianNetworks:
            raise ValueError('Allowed source names are {}, but {} was given'.format(
                self.BayesianNetworks, source_name))


    def validate_sample_count_string(self, sample_count_string):
        validation_regex = re.compile(r"^[0-9]+e[0-9]+$")
        result = validation_regex.match(sample_count_string)
        if result is None:
            raise ValueError("Incorrect format for sample count. E.g. 3e5.")


def get_algorithm_class(alg):
    if alg == 'iamb':
        return AlgorithmIAMB

    if alg == 'ipcmb':
        return AlgorithmIPCMB
