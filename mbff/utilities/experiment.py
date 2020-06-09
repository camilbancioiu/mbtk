import pickle
from pprint import pprint


class ExperimentalPathSet:

    def __init__(self, root):
        self.Root = root
        self.ExDsRepository = self.Root / 'exds_repository'
        self.ExpRunRepository = self.Root / 'exprun_repository'


    def __str__(self):
        max_name_width = max(map(len, self.__dict__.keys()))
        format_string = '{key:>' + str(max_name_width) + '}: {path}'

        keys = sorted(self.__dict__.keys())
        output = []
        output.append(format_string.format(key='Root', path=self.Root))
        for key in keys:
            if key == 'Root':
                continue
            path = self.shorten_path(key)
            output.append(format_string.format(key=key, path=path))
        return '\n'.join(output)


    def shorten_path(self, key):
        if key == 'Root':
            return self.Root
        path = self.__dict__[key]
        try:
            path = path.relative_to(self.Root)
        except ValueError:
            try:
                path = path.relative_to(self.Root.parent)
            except ValueError:
                pass
        return path



class ExperimentalSetup:

    def __init__(self):
        self.ExperimentDef = None
        self.ExDsDef = None
        self.Paths = None
        self.AlgorithmRunParameters = None
        self.Arguments = None


    def update_paths(self):
        self.Paths.Experiment = self.ExperimentDef.path
        self.Paths.ExDs = self.ExDsDef.path


    def filter_algruns(self):
        # Allow subclasses to implement arbitrary filtering in this method.
        pass


    def filter_algruns_by_tag(self, tag):
        self.AlgorithmRunParameters = self.get_algruns_by_tag(tag)


    def get_algruns_by_tag(self, tag):
        return [p for p in self.AlgorithmRunParameters if tag in p['tags']]


    def is_tag_present_in_any_algrun(self, tag):
        for parameters in self.AlgorithmRunParameters:
            if tag in parameters['tags']:
                return True
        return False



def configure_objects_subparser__paths(subparsers):
    subparser = subparsers.add_parser('paths')
    subparser.add_argument('verb', choices=['show'], default='show', nargs='?')



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
    subparser.add_argument('--type', type=str, default='',
                           help='lock type to use for `lock` and `unlock`')



def configure_objects_subparser__exp(subparsers):
    subparser = subparsers.add_parser('exp')
    subparser.add_argument('verb', choices=['show', 'run', 'lock', 'unlock', 'delete'],
                           default='show', nargs='?')
    subparser.add_argument('--index', type=str, default=None)
    subparser.add_argument('--type', type=str, default='',
                           help='lock type to use for `lock` and `unlock`')



def configure_objects_subparser__parameters(subparsers):
    subparser = subparsers.add_parser('parameters')
    subparser.add_argument('verb', choices=['list'], default='list', nargs='?')
    subparser.add_argument('--key', type=str, default=None)
    subparser.add_argument('--index', type=str, default=None)



def configure_objects_subparser__datapoints(subparsers):
    subparser = subparsers.add_parser('datapoints')
    subparser.add_argument('verb', choices=['list'], default='list', nargs='?')
    subparser.add_argument('--key', type=str, default=None)
    subparser.add_argument('--index', type=str, default=None)



def handle_command(arguments, experimental_setup):
    command_handled = False
    command_object = experimental_setup.Arguments.object
    command_verb = experimental_setup.Arguments.verb

    if command_object == 'paths':
        if command_verb == 'show':
            command_paths_show(experimental_setup)
            command_handled = True

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
            command_exds_delete(experimental_setup)
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

    elif command_object == 'parameters':
        if command_verb == 'list':
            command_parameters_list(experimental_setup)
            command_handled = True

    elif command_object == 'datapoints':
        if command_verb == 'list':
            command_datapoints_list(experimental_setup)
            command_handled = True

    return command_handled



def command_paths_show(experimental_setup):
    print(experimental_setup.Paths)



def command_exp_def_show(experimental_setup):
    view = experimental_setup.ExperimentDef.__dict__.copy()
    view['locks'] = experimental_setup.ExperimentDef.get_locks()
    view['folder_exists'] = experimental_setup.ExperimentDef.folder_exists()
    pprint(view)



def command_exds_def_show(experimental_setup):
    view = experimental_setup.ExDsDef.__dict__.copy()
    view['locks'] = experimental_setup.ExDsDef.get_locks()
    view['ready'] = experimental_setup.ExDsDef.exds_ready()
    view['folder_exists'] = experimental_setup.ExDsDef.folder_exists()
    pprint(view)



def command_exds_delete(experimental_setup):
    experimental_setup.ExDsDef.delete_folder()



def command_exds_show(experimental_setup):
    if experimental_setup.ExDsDef.exds_ready():
        ExDs = experimental_setup.ExDsDef.create_exds()
        ExDs.load()
        print(ExDs.info())
    else:
        print('Experimental Dataset must be built before information about it can be displayed.')



def command_exds_build(experimental_setup):
    if experimental_setup.ExDsDef.exds_ready():
        print('Experimental Dataset already built.')
    else:
        ExDs = experimental_setup.ExDsDef.create_exds()
        ExDs.build()
        print('Experimental Dataset has been built.')



def command_exds_unlock(experimental_setup):
    experimental_setup.ExDsDef.unlock_folder(experimental_setup.Arguments.type)



def command_exp_run(experimental_setup):
    selected_parameters = get_parameters_by_index(experimental_setup)
    experimental_setup.ExperimentDef.algorithm_run_parameters = selected_parameters
    Experiment = experimental_setup.ExperimentDef.create_experiment_run()
    Experiment.run()



def command_exp_unlock(experimental_setup):
    experimental_setup.ExperimentDef.unlock_folder(experimental_setup.Arguments.type)



def command_exp_lock(experimental_setup):
    experimental_setup.ExperimentDef.lock_folder(experimental_setup.Arguments.type)



def command_exds_lock(experimental_setup):
    experimental_setup.ExDsDef.lock_folder(experimental_setup.Arguments.type)



def command_exp_delete(experimental_setup):
    experimental_setup.ExperimentDef.delete_folder()



def command_parameters_show(experimental_setup):
    parameters_index = get_parameters_index(experimental_setup.Arguments.index)
    if parameters_index is None:
        parameters_list = experimental_setup.AlgorithmRunParameters
    else:
        parameters_list = experimental_setup.AlgorithmRunParameters[parameters_index]
    for i, parameters in enumerate(parameters_list):
        print('AlgorithmRun {} parameters:'.format(parameters_index.start + i))
        pprint(parameters)
        print()



def command_parameters_list(experimental_setup):
    specific_key = experimental_setup.Arguments.key
    parameters_index = get_parameters_index(experimental_setup)

    parameters_to_list = experimental_setup.AlgorithmRunParameters
    if parameters_index is not None:
        parameters_to_list = experimental_setup.AlgorithmRunParameters[parameters_index]

    for parameters in parameters_to_list:
        print('AlgorithmRun parameters [{}] {}:'.format(parameters['index'], parameters['ID']))
        if specific_key is not None:
            try:
                print('{}: {}'.format(specific_key, parameters[specific_key]))
            except KeyError:
                print('{}: {}'.format(specific_key, 'not found'))
        else:
            pprint(parameters)
        print()
    print('Total of', len(parameters_to_list), 'AlgorithmRun parameters')



def command_datapoints_list(experimental_setup):
    specific_key = experimental_setup.Arguments.key
    parameters_index = get_parameters_index(experimental_setup)

    parameters_to_list = experimental_setup.AlgorithmRunParameters
    if parameters_index is not None:
        parameters_to_list = experimental_setup.AlgorithmRunParameters[parameters_index]

    datapoints_folder = experimental_setup.ExperimentDef.subfolder('algorithm_run_datapoints')
    for parameters in parameters_to_list:
        datapoint_file = datapoints_folder / '{}.pickle'.format(parameters['ID'])
        try:
            with datapoint_file.open('rb') as f:
                datapoint = pickle.load(f)
        except FileNotFoundError:
            datapoint = None
        print('AlgorithmRun datapoint [{}] {}:'.format(parameters['index'], parameters['ID']))

        if datapoint is None:
            print('(missing)')
            print()
            continue

        if specific_key is not None:
            try:
                print('{}: {}'.format(specific_key, datapoint.__dict__[specific_key]))
            except KeyError:
                print('{}: {}'.format(specific_key, 'not found'))
        else:
            print(datapoint)
        print()

    print('Total: {} AlgorithmRun datapoints'.format(len(parameters_to_list)))



def get_parameters_by_index(experimental_setup):
    parameters_index = get_parameters_index(experimental_setup)
    if parameters_index is None:
        return experimental_setup.AlgorithmRunParameters
    else:
        return experimental_setup.AlgorithmRunParameters[parameters_index]



def get_parameters_index(experimental_setup):
    parameters_index = experimental_setup.Arguments.index
    if parameters_index is None:
        return None
    else:
        try:
            parameters_index = int(parameters_index)
            return slice(parameters_index, parameters_index + 1)
        except ValueError:
            if '-' in parameters_index:
                return create_slice_from_string(parameters_index)
            else:
                raise


def create_slice_from_string(s):
    parts = s.split('-')
    return slice(int(parts[0]), int(parts[1]) + 1)
