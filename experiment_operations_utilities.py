"""
A collection of functions required by multiple operations defined in the
:py:mod:`experiment_operations` module.
"""

from definitions import Experiments, ExperimentalDatasets, get_from_definitions

def map_over_experiment_definitions(experiment_names, opname, op, print_status=True):
    i = 1
    definitions = list(get_from_definitions(Experiments, experiment_names))
    results = []
    for definition in definitions:
        if opname != '' and print_status == True:
            print('{} Experiment {} ({} / {})'.format(opname, definition.name, i, len(definitions)))
        results.append(op(definition))
        i += 1
    return results


experiment_definition_table_format = '{:<30}\t{:<20}\t{:40}\t{:20}'

def experiment_definition_table_header():
    output = experiment_definition_table_format.format(
            'Experiment name', 'ExDs name', 'Parameters', 'Config')
    output += '\n' + experiment_definition_table_format.format(*(['--------------------']*4))
    return output

def experiment_definition_to_table_string(definition):
    parameters_description = ', '.join([
            '{}:{}'.format(pn, len(pv))
            for pn, pv in definition.parameters.items()
            ])
    config_description = ', '.join(sorted([
        '{}={}'.format(cn, cv)
        for cn, cv in definition.config.items()
        ]))

    return experiment_definition_table_format.format(
            definition.name, definition.exds_definition.name, parameters_description, config_description
            )

