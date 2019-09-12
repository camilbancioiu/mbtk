import pickle
from pprint import pprint

from mbff.utilities.Exceptions import CLICommandNotHandled


def handle_common_commands(command, arguments, ExperimentDef, ExdsDef, AlgorithmRunParameters):
    if command == 'show-exp-def':
        command_show_experiment_def(ExperimentDef)
    elif command == 'show-exds-def':
        command_show_exds_def(ExdsDef)
    elif command == 'build-exds':
        command_build_exds(arguments, ExdsDef)
    elif command == 'show-algruns':
        command_show_algruns(arguments, AlgorithmRunParameters)
    elif command == 'list-algruns':
        command_list_algrun(arguments, AlgorithmRunParameters)
    elif command == 'list-algrun-datapoints':
        command_list_algrun_datapoints(arguments, ExperimentDef)
    elif command == 'run-experiment':
        command_run_experiment(arguments, ExperimentDef, AlgorithmRunParameters)
    elif command == 'unlock-exp':
        command_unlock_experiment(arguments, ExperimentDef)
    elif command == 'lock-exp':
        command_lock_experiment(arguments, ExperimentDef)
    elif command == 'delete-exp':
        command_delete_exp(ExperimentDef)
    elif command == 'delete-exds':
        command_delete_exds(ExdsDef)
    elif command == 'lock-exds':
        command_lock_exds(arguments, ExdsDef)
    elif command == 'unlock-exds':
        command_unlock_exds(arguments, ExdsDef)
    elif command == 'show-exds':
        command_show_exds(arguments, ExdsDef)
    else:
        raise CLICommandNotHandled(command)



def command_show_experiment_def(ExperimentDef):
    view = ExperimentDef.__dict__.copy()
    view['locks'] = ExperimentDef.get_locks()
    view['folder_exists'] = ExperimentDef.folder_exists()
    pprint(view)



def command_show_exds_def(ExdsDef):
    view = ExdsDef.__dict__.copy()
    view['locks'] = ExdsDef.get_locks()
    view['ready'] = ExdsDef.exds_ready()
    view['folder_exists'] = ExdsDef.folder_exists()
    pprint(view)



def command_delete_exds(ExdsDef):
    ExdsDef.delete_folder()



def command_delete_exp(ExperimentDef):
    ExperimentDef.delete_folder()



def command_build_exds(arguments, ExdsDef):
    if ExdsDef.exds_ready():
        print('Experimental Dataset already built.')
    else:
        ExDs = ExdsDef.create_exds()
        ExDs.build()
        print('Experimental Dataset has been built.')



def command_show_exds(arguments, ExdsDef):
    if ExdsDef.exds_ready():
        ExDs = ExdsDef.create_exds()
        ExDs.load()
        print(ExDs.info())
    else:
        print('Experimental Dataset must be built before information about it can be displayed.')



def command_unlock_exds(arguments, ExdsDef):
    try:
        lock_type = arguments[0]
    except IndexError:
        lock_type = ''
    ExdsDef.unlock_folder(lock_type)



def command_lock_exds(arguments, ExdsDef):
    try:
        lock_type = arguments[0]
    except IndexError:
        lock_type = ''
    ExdsDef.lock_folder(lock_type)



def command_show_algruns(arguments, AlgorithmRunParameters):
    algrun_parameters_index = get_algrun_index(arguments)
    if algrun_parameters_index is None:
        algruns = AlgorithmRunParameters
    else:
        algruns = AlgorithmRunParameters[algrun_parameters_index]
    for i, algrun in enumerate(algruns):
        print('AlgorithmRun {} parameters:'.format(algrun_parameters_index.start + i))
        pprint(algrun)
        print()



def command_list_algrun(arguments, AlgorithmRunParameters):
    specific_key = None
    try:
        specific_key = arguments[0]
    except IndexError:
        pass

    for index, parameters in enumerate(AlgorithmRunParameters):
        print('AlgorithmRun {} parameters:'.format(index))
        if specific_key is not None:
            try:
                print('{}: {}'.format(specific_key, parameters[specific_key]))
            except KeyError:
                print('{}: {}'.format(specific_key, 'not found'))
        else:
            pprint(parameters)
        print()
    print('Total of', len(AlgorithmRunParameters), 'AlgorithmRun parameters')



def command_list_algrun_datapoints(arguments, ExperimentDef):
    datapoints_folder = ExperimentDef.subfolder('algorithm_run_datapoints')
    datapoint_files = sorted(list(datapoints_folder.iterdir()))

    algrun_parameters_index = get_algrun_index(arguments)
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



def command_run_experiment(arguments, ExperimentDef, AlgorithmRunParameters):
    ExperimentDef.algorithm_run_parameters = get_algruns_by_index(arguments, AlgorithmRunParameters)
    Experiment = ExperimentDef.create_experiment_run()
    Experiment.run()



def command_unlock_experiment(arguments, ExperimentDef):
    try:
        lock_type = arguments[0]
    except IndexError:
        lock_type = ''
    ExperimentDef.unlock_folder(lock_type)



def command_lock_experiment(arguments, ExperimentDef):
    try:
        lock_type = arguments[0]
    except IndexError:
        lock_type = ''
    ExperimentDef.lock_folder(lock_type)


def get_algruns_by_index(arguments, AlgorithmRunParameters):
    algrun_parameters_index = get_algrun_index(arguments)
    if algrun_parameters_index is None:
        return AlgorithmRunParameters
    else:
        return AlgorithmRunParameters[algrun_parameters_index]



def get_algrun_index(arguments):
    if len(arguments) == 0:
        return None
    else:
        algrun_parameters_index = arguments[0]
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
