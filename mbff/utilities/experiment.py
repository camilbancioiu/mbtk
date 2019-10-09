import pickle
from pprint import pprint


class ExperimentalPathSet:

    def __init__(self, root):
        self.Root = root
        self.ExDsRepository = self.Root / 'exds_repository'
        self.ExpRunRepository = self.Root / 'exprun_repository'



class ExperimentalSetup:

    def __init__(self):
        self.ExperimentDef = None
        self.ExDsDef = None
        self.Paths = None
        self.AlgorithmRunParameters = None
        self.Arguments = None


    def update_paths(self):
        self.Paths.Experiment = self.ExperimentDef.path


    def filter_algruns_by_tag(self, tag):
        self.AlgorithmRunParameters = [p for p in self.AlgorithmRunParameters if tag in p['tags']]


    def is_tag_present_in_any_algrun(self, tag):
        for parameters in self.AlgorithmRunParameters:
            if tag in parameters['tags']:
                return True
        return False



def configure_objects_subparser__exp_def(subparsers):
    subparser = subparsers.add_parser('exp-def')
    subparser.add_argument('verb', choices=['show'], default='show', nargs='?')



def configure_objects_subparser__exds_def(subparsers):
    subparser = subparsers.add_parser('exds-def')
    subparser.add_argument('verb', choices=['show'], default='show', nargs='?')



def configure_objects_subparser__exds(subparsers):
    subparser = subparsers.add_parser('exds')
    subparser.add_argument('verb', choices=['show', 'build', 'lock', 'unlock', 'delete'],
                           default='show', nargs='?')
    subparser.add_argument('--type', type=str, default='')



def configure_objects_subparser__exp(subparsers):
    subparser = subparsers.add_parser('exp')
    subparser.add_argument('verb', choices=['show', 'run', 'lock', 'unlock', 'delete'],
                           default='show', nargs='?')
    subparser.add_argument('--index', type=str, default=None)



def configure_objects_subparser__algruns(subparsers):
    subparser = subparsers.add_parser('algruns')
    subparser.add_argument('verb', choices=['show', 'list'],
                           default='show', nargs='?')
    subparser.add_argument('--key', type=str, default=None)


def configure_objects_subparser__algrun_datapoints(subparsers):
    subparser = subparsers.add_parser('algrun-datapoints')
    subparser.add_argument('verb', choices=['list'],
                           default='list', nargs='?')
    subparser.add_argument('--index', type=str, default=None)



def handle_command(experimental_setup):
    command_handled = False
    command_object = experimental_setup.Arguments.object
    command_verb = experimental_setup.Arguments.verb

    if command_object == 'exp-def':
        if command_verb == 'show':
            command_exp_def_show(experimental_setup)
            command_handled = True

    elif command_object == 'exds-def':
        if command_verb == 'show':
            command_exds_def_show(experimental_setup)
            command_handled = True

    elif command_object == 'exds':
        if command_verb == 'show':
            command_exds_show(experimental_setup)
            command_handled = True
        elif command_verb == 'build':
            command_exds_build(experimental_setup)
            command_handled = True
        elif command_verb == 'lock':
            command_exds_lock(experimental_setup)
            command_handled = True
        elif command_verb == 'unlock':
            command_exds_unlock(experimental_setup)
            command_handled = True
        elif command_verb == 'delete':
            command_exds_unlock(experimental_setup)
            command_handled = True

    elif command_object == 'exp':
        if command_verb == 'show':
            pass
        elif command_verb == 'run':
            command_exp_run(experimental_setup)
            command_handled = True
        elif command_verb == 'lock':
            command_exp_lock(experimental_setup)
            command_handled = True
        elif command_verb == 'unlock':
            command_exp_unlock(experimental_setup)
            command_handled = True
        elif command_verb == 'delete':
            command_exp_delete(experimental_setup)
            command_handled = True

    elif command_object == 'algruns':
        if command_verb == 'show':
            command_algruns_show(experimental_setup)
            command_handled = True
        elif command_verb == 'list':
            command_algruns_list(experimental_setup)
            command_handled = True

    elif command_object == 'algrun-datapoints':
        if command_verb == 'list':
            command_algrun_datapoints_list(experimental_setup)
            command_handled = True

    return command_handled



def command_exp_def_show(experimental_setup):
    view = experimental_setup.ExperimentDef.__dict__.copy()
    view['locks'] = experimental_setup.ExperimentDef.get_locks()
    view['folder_exists'] = experimental_setup.ExperimentDef.folder_exists()
    pprint(view)



def command_exds_def_show(experimental_setup):
    view = experimental_setup.ExdsDef.__dict__.copy()
    view['locks'] = experimental_setup.ExdsDef.get_locks()
    view['ready'] = experimental_setup.ExdsDef.exds_ready()
    view['folder_exists'] = experimental_setup.ExdsDef.folder_exists()
    pprint(view)



def command_exds_delete(experimental_setup):
    experimental_setup.ExdsDef.delete_folder()



def command_exds_build(experimental_setup):
    if experimental_setup.ExdsDef.exds_ready():
        print('Experimental Dataset already built.')
    else:
        ExDs = experimental_setup.ExdsDef.create_exds()
        ExDs.build()
        print('Experimental Dataset has been built.')



def command_exds_show(experimental_setup):
    if experimental_setup.ExdsDef.exds_ready():
        ExDs = experimental_setup.ExdsDef.create_exds()
        ExDs.load()
        print(ExDs.info())
    else:
        print('Experimental Dataset must be built before information about it can be displayed.')



def command_exds_unlock(experimental_setup):
    experimental_setup.ExdsDef.unlock_folder(experimental_setup.Arguments.type)



def command_exp_run(experimental_setup):
    selected_algruns = get_algruns_by_index(experimental_setup.Arguments, experimental_setup.AlgorithmRunParameters)
    experimental_setup.ExperimentDef.algorithm_run_parameters = selected_algruns
    Experiment = experimental_setup.ExperimentDef.create_experiment_run()
    Experiment.run()



def command_exp_unlock(experimental_setup):
    experimental_setup.ExperimentDef.unlock_folder(experimental_setup.Arguments.type)



def command_exp_lock(experimental_setup):
    experimental_setup.ExperimentDef.lock_folder(experimental_setup.Arguments.type)



def command_exds_lock(experimental_setup):
    experimental_setup.ExdsDef.lock_folder(experimental_setup.Arguments.type)



def command_exp_delete(experimental_setup):
    experimental_setup.ExperimentDef.delete_folder()



def command_algruns_show(experimental_setup):
    algrun_parameters_index = get_algrun_index(experimental_setup.Arguments.index)
    if algrun_parameters_index is None:
        algruns = experimental_setup.AlgorithmRunParameters
    else:
        algruns = experimental_setup.AlgorithmRunParameters[algrun_parameters_index]
    for i, algrun in enumerate(algruns):
        print('AlgorithmRun {} parameters:'.format(algrun_parameters_index.start + i))
        pprint(algrun)
        print()



def command_algruns_list(experimental_setup):
    specific_key = experimental_setup.Arguments.key
    for index, parameters in enumerate(experimental_setup.AlgorithmRunParameters):
        print('AlgorithmRun {} parameters:'.format(index))
        if specific_key is not None:
            try:
                print('{}: {}'.format(specific_key, parameters[specific_key]))
            except KeyError:
                print('{}: {}'.format(specific_key, 'not found'))
        else:
            pprint(parameters)
        print()
    print('Total of', len(experimental_setup.AlgorithmRunParameters), 'AlgorithmRun parameters')



def command_algrun_datapoints_list(experimental_setup):
    datapoints_folder = experimental_setup.ExperimentDef.subfolder('algorithm_run_datapoints')
    datapoint_files = sorted(list(datapoints_folder.iterdir()))

    algrun_parameters_index = get_algrun_index(experimental_setup.Arguments.index)
    if algrun_parameters_index is None:
        datapoint_files_to_list = datapoint_files
    else:
        datapoint_files_to_list = datapoint_files[algrun_parameters_index]

    for datapoint_file in datapoint_files_to_list:
        with datapoint_file.open('rb') as f:
            datapoint = pickle.load(f)
        print(datapoint)
        print()

    print('Total: {} AlgorithmRun datapoints'.format(len(datapoint_files_to_list)))



def get_algruns_by_index(experimental_setup):
    algrun_parameters_index = get_algrun_index(experimental_setup.Arguments)
    if algrun_parameters_index is None:
        return experimental_setup.AlgorithmRunParameters
    else:
        return experimental_setup.AlgorithmRunParameters[algrun_parameters_index]



def get_algrun_index(experimental_setup):
    algrun_parameters_index = experimental_setup.Arguments.index
    if algrun_parameters_index is None:
        return None
    else:
        try:
            algrun_parameters_index = int(algrun_parameters_index)
            return slice(algrun_parameters_index, algrun_parameters_index + 1)
        except ValueError:
            if '-' in algrun_parameters_index:
                return create_slice_from_string(algrun_parameters_index)
            else:
                raise


def create_slice_from_string(s):
    parts = s.split('-')
    return slice(int(parts[0]), int(parts[1]) + 1)
