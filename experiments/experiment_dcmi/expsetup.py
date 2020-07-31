import re
import pickle
import mbff.math.Variable
import mbff.utilities.experiment as util


class DCMIEvExpPathSet(util.ExperimentalPathSet):

    def __init__(self, root):
        super().__init__(root)
        self.BIFRepository = self.Root / 'bif_repository'



class DCMIEvExpSetup(util.ExperimentalSetup):

    def __init__(self):
        super().__init__()
        self.DatasetName = None
        self.Omega = None
        self.CITest_Significance = None
        self.SampleCountString = None
        self.SampleCount = None
        self.AllowedDatasetNames = ['alarm', 'pathfinder', 'andes', 'munin']
        self.AllowedADTreeTypes = ['static', 'dynamic']
        self.AllowedLLT = ['0', '5', '10']
        self.DefaultTags = ['unoptimized', 'adtree-static-llt0',
                            'adtree-static-llt5', 'adtree-static-llt10',
                            'adtree-dynamic-llt0', 'adtree-dynamic-llt5',
                            'adtree-dynamic-llt10', 'dcmi']


    def set_arguments(self, arguments):
        self.validate_arguments(arguments)

        self.Arguments = arguments
        self.DatasetName = self.Arguments.dataset_name
        self.SampleCountString = self.Arguments.sample_count
        self.SampleCount = int(float(self.SampleCountString))
        self.Omega = mbff.math.Variable.Omega(self.SampleCount)


    def calculate_absolute_LLT(self, llt):
        return int(self.SampleCount * int(llt) / 1000)


    def update_paths(self):
        super().update_paths()
        self.Paths.ADTreeRepository = self.Paths.Experiment / 'adtrees'
        self.Paths.ADTreeRepository.mkdir(parents=True, exist_ok=True)

        self.Paths.JHTRepository = self.Paths.Experiment / 'jht'
        self.Paths.JHTRepository.mkdir(parents=True, exist_ok=True)

        self.Paths.DoFCacheRepository = self.Paths.Experiment / 'dof_cache'
        self.Paths.DoFCacheRepository.mkdir(parents=True, exist_ok=True)

        self.Paths.CITestResultRepository = self.Paths.Experiment / 'ci_test_results'
        self.Paths.CITestResultRepository.mkdir(parents=True, exist_ok=True)


    def filter_algruns(self):
        targets_to_skip = None
        if self.DatasetName == 'pathfinder':
            targets_to_skip = [0]

        if targets_to_skip is None:
            return

        for parameters in self.AlgorithmRunParameters:
            if parameters['target'] in targets_to_skip:
                parameters['skip'] = True


    def get_ADTree_path(self, tree_type, llt):
        adtree_filename = 'adtree_{}_llt{}.pickle'.format(tree_type, llt)
        return self.Paths.ADTreeRepository / adtree_filename


    def preload_ADTrees(self):
        preloaded_adtrees = dict()
        for parameters in self.AlgorithmRunParameters:
            if 'adtree' in parameters['tags']:
                tree_path = parameters.get('ci_test_ad_tree_path__load', None)
                if tree_path is not None:
                    adtree = None
                    try:
                        adtree = preloaded_adtrees[tree_path]
                    except KeyError:
                        with tree_path.open('rb') as f:
                            adtree = pickle.load(f)
                        preloaded_adtrees[tree_path] = adtree
                    parameters['ci_test_ad_tree_preloaded'] = adtree
                    del parameters['ci_test_ad_tree_path__load']


    def validate_arguments(self, arguments):
        self.validate_dataset_name(arguments.dataset_name)
        self.validate_sample_count_string(arguments.sample_count)


    def validate_dataset_name(self, dataset_name):
        if dataset_name not in self.AllowedDatasetNames:
            raise ValueError('Allowed dataset names are {}, but {} was given'.format(
                self.AllowedDatasetNames, dataset_name))


    def validate_sample_count_string(self, sample_count_string):
        validation_regex = re.compile(r"^[0-9]+e[0-9]+$")
        result = validation_regex.match(sample_count_string)
        if result is None:
            raise ValueError("Incorrect format for sample count. E.g. 3e5.")
