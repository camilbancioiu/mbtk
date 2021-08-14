import re

import mbtk.utilities.experiment as util
from mbtk.structures.BayesianNetwork import BayesianNetwork

from mbtk.algorithms.mb.iamb import AlgorithmIAMB
from mbtk.algorithms.mb.ipcmb import AlgorithmIPCMB
from mbtk.dataset.ExperimentalDataset import ExperimentalDataset
from mbtk.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbtk.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
from mbtk.experiment.AlgorithmRun import AlgorithmRun
from mbtk.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbtk.experiment.ExperimentDefinition import ExperimentDefinition
from mbtk.experiment.ExperimentRun import ExperimentRun
from mbtk.math.BNCorrelationEstimator import BNCorrelationEstimator
from mbtk.math.DSeparationCITest import DSeparationCITest
from mbtk.math.G_test__unoptimized import G_test
from mbtk.math.DoFCalculators import StructuralDoF


SOURCES = ['bn', 'ds']
BAYESIAN_NETWORKS = ['alarm', 'mildew', 'child']
ALGORITHMS = ['iamb', 'ipcmb']


class DsepAlgsEvalExpPathSet(util.ExperimentalPathSet):

    def __init__(self, root):
        super().__init__(root)
        self.BIFRepository = self.Root / 'bif_repository'
        self.Datapoints = None
        self.CITestResultRepository = None
        self.HResultRepository = None
        self.Summaries = None
        self.Plots = None



class DsepAlgsEvalExpSetup(util.ExperimentalSetup):

    def __init__(self, root, arguments):
        self.validate_arguments(arguments)
        super().__init__()
        self.root = root
        self.arguments = arguments
        self.paths = DsepAlgsEvalExpPathSet(root)
        self.algorithm_name = ""
        self.algorithm_class = None
        self.bayesian_network_name = ""
        self.bayesian_network = None
        self.source_type = ""
        self.sample_count = None
        self.sample_count_string = ""
        self.ci_test_significance = None

        self.setup(arguments)
        self.define_experiment()
        self.define_source()
        self.update_paths()
        self.create_algrun_parameters()


    def setup(self, arguments):
        self.Arguments = arguments
        self.algorithm_name = self.arguments.algorithm
        self.algorithm_class = self.get_algorithm_class()

        self.source_type = self.arguments.source_type

        self.bayesian_network_name = self.arguments.source_name
        bn_sourcepath = self.paths.BIFRepository / self.bayesian_network_name
        bn_sourcepath = bn_sourcepath.with_suffix('.bif')

        self.bayesian_network = BayesianNetwork.from_bif_file(bn_sourcepath, use_cache=True)
        self.bayesian_network.finalize()

        if self.source_type == 'ds':
            if self.arguments.sample_count is None:
                raise ValueError('sample count required')
            self.sample_count_string = self.arguments.sample_count
            self.sample_count = int(float(self.sample_count_string))
            self.ci_test_significance = 0.95


    def update_paths(self):
        self.Paths = self.paths
        super().update_paths()
        self.paths.Datapoints = self.experiment_definition.subfolder('algorithm_run_datapoints')
        self.paths.CITestResultRepository = self.experiment_definition.subfolder('ci_test_results')
        self.paths.HResultRepository = self.experiment_definition.subfolder('heuristic_results')
        self.paths.Summaries = self.experiment_definition.subfolder('summaries')
        self.paths.Plots = self.experiment_definition.subfolder('plots')


    def define_experiment(self):
        experiment_name = 'DsepAlgsEval'
        definition = ExperimentDefinition(
            self.paths.ExpRunRepository,
            experiment_name,
            self.create_experiment_run_stem(self.source_type))

        definition.experiment_run_class = ExperimentRun
        definition.algorithm_run_class = AlgorithmRun
        definition.algorithm_run_configuration = {
            'algorithm': self.algorithm_class
        }
        definition.algorithm_run_datapoint_class = AlgorithmRunDatapoint
        definition.save_algorithm_run_datapoints = True

        self.experiment_definition = definition
        self.ExperimentDef = self.experiment_definition


    def define_source(self):
        if self.source_type != 'ds':
            return None

        bayesian_network = self.bayesian_network_name
        sample_count = self.sample_count
        sample_count_str = self.sample_count_string
        exds_name = 'synthetic_{}_{}'.format(bayesian_network, sample_count_str)

        definition = ExperimentalDatasetDefinition(self.paths.ExDsRepository, exds_name)
        definition.exds_class = ExperimentalDataset
        definition.source = SampledBayesianNetworkDatasetSource
        definition.source_configuration = {
            'sourcepath': self.paths.BIFRepository / (bayesian_network + '.bif'),
            'sample_count': sample_count,
            'random_seed': 81628965211 + sample_count
        }

        self.exds_definition = definition
        self.experiment_definition.exds_definition = self.exds_definition
        self.ExDsDef = self.exds_definition


    def create_algrun_parameters(self):
        bayesian_network = self.bayesian_network
        citrrepo = self.paths.CITestResultRepository
        hrrepo = self.paths.HResultRepository

        ci_test_class = self.get_ci_test_class()
        variables = list(range(len(bayesian_network)))
        parameters_list = list()
        for target in variables:
            parameters = {
                'target': target,
                'all_variables': variables,
                'source_bayesian_network': bayesian_network,
                'ci_test_class': ci_test_class,
                'correlation_heuristic_class': BNCorrelationEstimator,
                'ci_test_dof_calculator_class': StructuralDoF,
                'ci_test_significance': self.ci_test_significance,
                'tags': [
                    self.algorithm_name,
                    self.bayesian_network_name
                ],
            }

            ID = self.create_algrun_ID(parameters, self.source_type)
            parameters['ID'] = 'run_' + ID
            citr_filename = 'citr_' + ID + '.pickle'
            parameters['ci_test_results_path__save'] = citrrepo / citr_filename
            hr_filename = 'hr_' + ID + '.pickle'
            parameters['heuristic_results_path__save'] = hrrepo / hr_filename
            parameters_list.append(parameters)

        for index, parameters in enumerate(parameters_list):
            parameters['index'] = index

        self.algorithm_run_parameters = parameters_list
        self.AlgorithmRunParameters = parameters_list



    def create_experiment_run_stem(self, source_type):
        alg = self.algorithm_name
        sample_count = self.sample_count_string
        bayesian_network = self.bayesian_network_name

        if source_type == 'bn':
            return f'{source_type}_{bayesian_network}_{alg}'

        if source_type == 'ds':
            return f'{source_type}_{bayesian_network}_{sample_count}_{alg}'


    def create_algrun_ID(self, parameters, source_type):
        target = parameters['target']
        if source_type == 'bn':
            ID = 'bn_{}_{}_T{}'.format(
                self.bayesian_network_name,
                self.algorithm_name,
                target)
        if source_type == 'ds':
            ID = 'ds_{}_{}_{}_T{}'.format(
                self.bayesian_network_name,
                self.sample_count_string,
                self.algorithm_name,
                target)

        return ID


    def validate_arguments(self, arguments):
        self.validate_algorithm(arguments.algorithm)
        self.validate_source_type(arguments.source_type)
        self.validate_source_name(arguments.source_name)
        if arguments.source_type == ['ds']:
            self.validate_sample_count_string(arguments.sample_count)


    def validate_algorithm(self, algorithm_name):
        if algorithm_name not in ALGORITHMS:
            raise ValueError(f'Algorithm {algorithm_name} unknown')


    def validate_source_type(self, source_type):
        if source_type not in SOURCES:
            error_message = 'Allowed source types are {}, but {} was given'
            error_message = error_message.format(SOURCES, source_type)
            raise ValueError(error_message)


    def validate_source_name(self, source_name):
        if source_name not in BAYESIAN_NETWORKS:
            error_message = 'Allowed source names are {}, but {} was given'
            error_message = error_message.format(BAYESIAN_NETWORKS, source_name)
            raise ValueError(error_message)


    def validate_sample_count_string(self, sample_count_string):
        validation_regex = re.compile(r"^[0-9]+e[0-9]+$")
        result = validation_regex.match(sample_count_string)
        if result is None:
            raise ValueError("Incorrect format for sample count. E.g. 3e5.")


    def make_bn_datapoints_path(self):
        bn_definition = self.make_bn_experiment_definition()
        return bn_definition.subfolder('algorithm_run_datapoints')


    def make_bn_experiment_definition(self):
        return ExperimentDefinition(
            self.paths.ExpRunRepository,
            self.experiment_definition.name,
            self.create_experiment_run_stem('bn'))


    def get_ci_test_class(self):
        if self.source_type == 'bn':
            return DSeparationCITest
        if self.source_type == 'ds':
            return G_test


    def get_algorithm_class(self):
        if self.algorithm_name == 'iamb':
            return AlgorithmIAMB
        if self.algorithm_name == 'ipcmb':
            return AlgorithmIPCMB
