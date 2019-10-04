import pickle
from pprint import pprint


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



class CommandContext:

    def __init__(self, args=None, expdef=None, exdsdef=None, algrunparams=None):
        self.arguments = args
        self.ExperimentDef = expdef
        self.ExdsDef = exdsdef
        self.AlgorithmRunParameters = algrunparams



def handle_command(command_context):
    command_handled = False
    command_object = command_context.arguments.object
    command_verb = command_context.arguments.verb

    if command_object == 'exp-def':
        if command_verb == 'show':
            command_exp_def_show(command_context)
            command_handled = True

    elif command_object == 'exds-def':
        if command_verb == 'show':
            command_exds_def_show(command_context)
            command_handled = True

    elif command_object == 'exds':
        if command_verb == 'show':
            command_exds_show(command_context)
            command_handled = True
        elif command_verb == 'build':
            command_exds_build(command_context)
            command_handled = True
        elif command_verb == 'lock':
            command_exds_lock(command_context)
            command_handled = True
        elif command_verb == 'unlock':
            command_exds_unlock(command_context)
            command_handled = True
        elif command_verb == 'delete':
            command_exds_unlock(command_context)
            command_handled = True

    elif command_object == 'exp':
        if command_verb == 'show':
            pass
        elif command_verb == 'run':
            command_exp_run(command_context)
            command_handled = True
        elif command_verb == 'lock':
            command_exp_lock(command_context)
            command_handled = True
        elif command_verb == 'unlock':
            command_exp_unlock(command_context)
            command_handled = True
        elif command_verb == 'delete':
            command_exp_delete(command_context)
            command_handled = True

    elif command_object == 'algruns':
        if command_verb == 'show':
            command_algruns_show(command_context)
            command_handled = True
        elif command_verb == 'list':
            command_algruns_list(command_context)
            command_handled = True

    elif command_object == 'algrun-datapoints':
        if command_verb == 'list':
            command_algrun_datapoints_list(command_context)
            command_handled = True

    return command_handled



def command_exp_def_show(command_context):
    view = command_context.ExperimentDef.__dict__.copy()
    view['locks'] = command_context.ExperimentDef.get_locks()
    view['folder_exists'] = command_context.ExperimentDef.folder_exists()
    pprint(view)



def command_exds_def_show(command_context):
    view = command_context.ExdsDef.__dict__.copy()
    view['locks'] = command_context.ExdsDef.get_locks()
    view['ready'] = command_context.ExdsDef.exds_ready()
    view['folder_exists'] = command_context.ExdsDef.folder_exists()
    pprint(view)



def command_exds_delete(command_context):
    command_context.ExdsDef.delete_folder()



def command_exds_build(command_context):
    if command_context.ExdsDef.exds_ready():
        print('Experimental Dataset already built.')
    else:
        ExDs = command_context.ExdsDef.create_exds()
        ExDs.build()
        print('Experimental Dataset has been built.')



def command_exds_show(command_context):
    if command_context.ExdsDef.exds_ready():
        ExDs = command_context.ExdsDef.create_exds()
        ExDs.load()
        print(ExDs.info())
    else:
        print('Experimental Dataset must be built before information about it can be displayed.')



def command_exds_unlock(command_context):
    command_context.ExdsDef.unlock_folder(command_context.arguments.type)



def command_exp_run(command_context):
    selected_algruns = get_algruns_by_index(command_context.arguments, command_context.AlgorithmRunParameters)
    command_context.ExperimentDef.algorithm_run_parameters = selected_algruns
    Experiment = command_context.ExperimentDef.create_experiment_run()
    Experiment.run()



def command_exp_unlock(command_context):
    command_context.ExperimentDef.unlock_folder(command_context.arguments.type)



def command_exp_lock(command_context):
    command_context.ExperimentDef.lock_folder(command_context.arguments.type)



def command_exds_lock(command_context):
    command_context.ExdsDef.lock_folder(command_context.arguments.type)



def command_exp_delete(command_context):
    command_context.ExperimentDef.delete_folder()



def command_algruns_show(command_context):
    algrun_parameters_index = get_algrun_index(command_context.arguments.index)
    if algrun_parameters_index is None:
        algruns = command_context.AlgorithmRunParameters
    else:
        algruns = command_context.AlgorithmRunParameters[algrun_parameters_index]
    for i, algrun in enumerate(algruns):
        print('AlgorithmRun {} parameters:'.format(algrun_parameters_index.start + i))
        pprint(algrun)
        print()



def command_algruns_list(command_context):
    specific_key = command_context.arguments.key
    for index, parameters in enumerate(command_context.AlgorithmRunParameters):
        print('AlgorithmRun {} parameters:'.format(index))
        if specific_key is not None:
            try:
                print('{}: {}'.format(specific_key, parameters[specific_key]))
            except KeyError:
                print('{}: {}'.format(specific_key, 'not found'))
        else:
            pprint(parameters)
        print()
    print('Total of', len(command_context.AlgorithmRunParameters), 'AlgorithmRun parameters')



def command_algrun_datapoints_list(command_context):
    datapoints_folder = command_context.ExperimentDef.subfolder('algorithm_run_datapoints')
    datapoint_files = sorted(list(datapoints_folder.iterdir()))

    algrun_parameters_index = get_algrun_index(command_context.arguments.index)
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



def get_algruns_by_index(command_context):
    algrun_parameters_index = get_algrun_index(command_context.arguments)
    if algrun_parameters_index is None:
        return command_context.AlgorithmRunParameters
    else:
        return command_context.AlgorithmRunParameters[algrun_parameters_index]



def get_algrun_index(command_context):
    algrun_parameters_index = command_context.arguments.index
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
