import sys
import re
import os
import pickle
from pathlib import Path


# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd()).parents[0]
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))

import mbff.utilities.experiment as util


class CustomExperimentalPathSet(util.ExperimentalPathSet):

    def __init__(self, root):
        super().__init__(root)
        self.BIFRepository = self.Root / 'bif_repository'



class CustomExperimentalSetup(util.ExperimentalSetup):

    def __init__(self):
        super().__init__()
        self.Omega = None
        self.CITest_Significance = None
        self.LLT = None
        self.ADTree = None


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

        adtree_filename = 'adtree_{}_llt{}.pickle'.format(self.ExDsDef.name, self.LLT)
        self.Paths.ADTree = self.Paths.ADTreeRepository / adtree_filename


    def preload_ADTree(self):
        with self.Paths.ADTree.open('rb') as f:
            self.ADTree = pickle.load(f)
        self.set_preloaded_ADTree_to_relevant_algrun_parameters()


    def set_preloaded_ADTree_to_relevant_algrun_parameters(self):
        for parameters in self.AlgorithmRunParameters:
            if 'adtree' in parameters['tags']:
                parameters['ci_test_ad_tree_preloaded'] = self.ADTree
                del parameters['ci_test_ad_tree_path__load']



def validate_sample_count_string(sample_count_string):
    validation_regex = re.compile(r"^[0-9]+e[0-9]+$")
    result = validation_regex.match(sample_count_string)
    if result is None:
        raise ValueError("Incorrect format for sample count. E.g. 3e5.")


################################################################################
# Command-line interface
if __name__ == '__main__':
    import definitions
    import algrun_parameters
    import commands

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--preload-adtree', action='store_true')
    argparser.add_argument('--algrun-tag', type=str, default=None, nargs='?')

    argparser.add_argument('sample_count', type=str, default=None)

    object_subparsers = argparser.add_subparsers(dest='object')
    object_subparsers.required = True

    util.configure_objects_subparser__paths(object_subparsers)
    util.configure_objects_subparser__exp_def(object_subparsers)
    util.configure_objects_subparser__exds_def(object_subparsers)
    util.configure_objects_subparser__exds(object_subparsers)
    util.configure_objects_subparser__exp(object_subparsers)
    util.configure_objects_subparser__parameters(object_subparsers)
    util.configure_objects_subparser__datapoints(object_subparsers)

    commands.configure_objects_subparser__adtree(object_subparsers)

    arguments = argparser.parse_args()

    validate_sample_count_string(arguments.sample_count)

    experimental_setup = CustomExperimentalSetup()
    experimental_setup.Arguments = arguments
    experimental_setup.CITest_Significance = 0.95
    experimental_setup.LLT = 0
    experimental_setup.Paths = CustomExperimentalPathSet(EXPERIMENTS_ROOT)
    experimental_setup.ExDsDef = definitions.exds_definition(experimental_setup)
    experimental_setup.ExperimentDef = definitions.experiment_definition(experimental_setup)
    experimental_setup.update_paths()
    experimental_setup.AlgorithmRunParameters = algrun_parameters.create_algrun_parameters(experimental_setup)

    if arguments.algrun_tag is not None:
        experimental_setup.filter_algruns_by_tag(arguments.algrun_tag)

    if arguments.preload_adtree is True:
        if experimental_setup.is_tag_present_in_any_algrun('adtree'):
            experimental_setup.preload_ADTree()


    command_handled = util.handle_command(experimental_setup)
    if command_handled is False:
        commands.handle_command(experimental_setup)
