"""
A collection of functions required by multiple operations defined in the
:py:mod:`exds_operations` module.
"""

import experimental_dataset

def map_over_exds_definitions(exds_names, opname, op, print_status=True):
    """
    Apply the callback ``op`` on the specified ExDs.

    This function collects the definitions of the ExDs specified in
    ``exds_names``, then calls ``op(definition)`` for all of them, while collecting
    the return values of each call.
    """
    i = 1
    definitions = list(get_from_definitions(ExperimentalDatasets, exds_names))
    results = []
    for definition in definitions:
        if opname != '' and print_status == True:
            mpprint('{} ExDs {} ({} / {})'.format(opname, definition.name, i, len(definitions)))
        results.append(op(definition))
        i += 1
    return results

def map_over_exds_definitions_parallel(exds_names, opname, op, print_status=True):
    """
    Apply the callback ``op`` on the specified ExDs *in parallel*.

    This function collects the definitions of the ExDs specified in
    ``exds_names``, then calls ``op(definition)`` for all of them, while collecting
    the return values of each call. The individual calls of ``op(definition)``
    are performed by worker-processes. Process handling is done using
    :py:class:`multiprocessing.Pool`.

    The actual function run by the worker-processes is
    :py:func:`map_over_exds_definitions_single`.

    The number of worker-processes spawned is controlled by the option
    ``--parallelism``.
    """
    definitions = list(get_from_definitions(ExperimentalDatasets, exds_names))
    parallel_arguments = zip(
        definitions,
        range(1, len(definitions) + 1), 
        itertools.repeat(op),
        itertools.repeat(opname),
        itertools.repeat(print_status),
        itertools.repeat(len(definitions))
        )
    results = []
    with multiprocessing.Pool(parallelism) as pool:
        results = pool.map(map_over_exds_definitions_single, parallel_arguments)
    return results

def map_over_exds_definitions_single(args):
    """
    Apply an ExDs operation callback on a single ExDs definition.

    This is the function run by the worker-processes spawned by
    :py:func:`map_over_exds_definitions_parallel`.
    """
    (definition, index, op, opname, print_status, len_definitions) = args
    if opname != '' and print_status == True:
        mpprint('{} ExDs {} ({} / {})'.format(opname, definition.name, index, len_definitions))
    return op(definition)

def exds_definition_table_format_string():
    """Return a format string for a single row of the table printed by :py:func:`exds_operations.op_list`."""
    return '{0:<30}\t{1:<12}\t{2:<12}\t{3!s:<10}\t{4:<12}\t{5:<18}\t{6}'

def exds_definition_table_header():
    """Return a formatted string containing the header of the table printed by :py:func:`exds_operations.op_list`."""
    return exds_definition_table_format_string().format('ExDs name', 'Industry', 'Train rows', 'Trim freqs', 'Folder', 'Tags', 'Locks')

def exds_definition_to_table_string(definition):
    """Return a row representing the ExDs ``definition`` in the table printed by :py:func:`exds_operations.op_list`."""
    locks = filter(definition.folder_is_locked, experimental_dataset.lock_types)
    return exds_definition_table_format_string().format(
            definition.name, definition.industry, definition.train_rows_proportion,
            definition.trim_freqs, str(definition.folder_exists()), ', '.join(definition.tags), ', '.join(locks))
