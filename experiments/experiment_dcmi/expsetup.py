import re
import pickle
import mbff.math.Variable
import mbff.utilities.experiment as util


class CustomExperimentalPathSet(util.ExperimentalPathSet):

    def __init__(self, root):
        super().__init__(root)
        self.BIFRepository = self.Root / 'bif_repository'



class CustomExperimentalSetup(util.ExperimentalSetup):

    def __init__(self):
        super().__init__()
        self.DatasetName = None
        self.Omega = None
        self.CITest_Significance = None
        self.LLT = None
        self.ADTree = None
        self.SampleCountString = None
        self.SampleCount = None
        self.AllowedDatasetNames = ['alarm', 'pathfinder', 'andes']
        self.AllowedLLTArgument = [0, 5, 10]
        self.DefaultTags = ['unoptimized', 'adtree-llt0', 'adtree-llt5', 'adtree-llt10', 'dcmi']


    def set_arguments(self, arguments):
        self.validate_arguments(arguments)

        self.Arguments = arguments
        self.DatasetName = self.Arguments.dataset_name
        self.SampleCountString = self.Arguments.sample_count
        self.SampleCount = int(float(self.SampleCountString))

        self.LLTArgument = self.Arguments.llt
        self.LLT = self.calculate_absolute_LLT_from_llt_argument(self.LLTArgument)

        self.Omega = mbff.math.Variable.Omega(self.SampleCount)


    def calculate_absolute_LLT_from_llt_argument(self, llt):
        return int(self.SampleCount * llt / 1000)


    def update_paths(self):
        super().update_paths()
        self.Paths.ADTreeRepository = self.Paths.Experiment / 'adtrees'
        self.Paths.JHTRepository = self.Paths.Experiment / 'jht'
        self.Paths.CITestResultRepository = self.Paths.Experiment / 'ci_test_results'
        self.Paths.DoFCacheRepository = self.Paths.Experiment / 'dof_cache'

        self.Paths.ADTreeRepository.mkdir(parents=True, exist_ok=True)
        self.Paths.JHTRepository.mkdir(parents=True, exist_ok=True)
        self.Paths.CITestResultRepository.mkdir(parents=True, exist_ok=True)
        self.Paths.DoFCacheRepository.mkdir(parents=True, exist_ok=True)

        self.Paths.ADTree = self.get_ADTree_path_for_llt_argument(self.LLTArgument)


    def get_ADTree_path_for_llt_argument(self, llt):
        adtree_filename = 'adtree_{}_llt{}.pickle'.format(self.ExDsDef.name, llt)
        return self.Paths.ADTreeRepository / adtree_filename


    def preload_ADTree(self):
        import gc
        print('Starting AD-tree preloading...')
        with self.Paths.ADTree.open('rb') as f:
            self.ADTree = pickle.load(f)
        gc.collect()
        print('AD-tree preloading complete.')
        self.set_preloaded_ADTree_to_relevant_algrun_parameters()


    def set_preloaded_ADTree_to_relevant_algrun_parameters(self):
        for parameters in self.AlgorithmRunParameters:
            if 'adtree' in parameters['tags']:
                parameters['ci_test_ad_tree_preloaded'] = self.ADTree
                del parameters['ci_test_ad_tree_path__load']


    def validate_arguments(self, arguments):
        self.validate_dataset_name(arguments.dataset_name)
        self.validate_sample_count_string(arguments.sample_count)
        self.validate_llt(arguments.llt)


    def validate_dataset_name(self, dataset_name):
        if dataset_name not in self.AllowedDatasetNames:
            raise ValueError('Allowed dataset names are {}, but {} was given'.format(
                self.AllowedDatasetNames, dataset_name))


    def validate_sample_count_string(self, sample_count_string):
        validation_regex = re.compile(r"^[0-9]+e[0-9]+$")
        result = validation_regex.match(sample_count_string)
        if result is None:
            raise ValueError("Incorrect format for sample count. E.g. 3e5.")


    def validate_llt(self, llt):
        if llt not in self.AllowedLLTArgument:
            raise ValueError('Allowed values for the --llt argument are {}'.format(self.AllowedLLTArgument))
