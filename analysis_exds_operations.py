import itertools
import functools as F
import utilities as util
from pathlib import Path
import sys
from pprint import pformat
import csv
from definitions import Experiments, ExperimentalDatasets, get_from_definitions
from experimental_dataset import *
import pickle
import datetime
import numpy

arguments = None

### Analysis-exds operation 'print-table-diff'
def op_print_table_diff(exds_names):
    if arguments.diff_table_tag == '':
        print('A value for --diff-table-tag is required.')
        return

    map_over_exds_definitions(exds_names, 'Print table diff',
            op_print_table_diff_single)

def op_print_table_diff_single(definition):
    if definition.folder_exists():
        exds = ExperimentalDataset(definition)
        print('Loading ExDs {}... '.format(definition.name), end='')
        exds.load()
        print('Done.')

        c = len(list(exds.topics))
        folder = definition.folder + '/ks'
        dt = arguments.diff_table_tag
        for t in range(c):
            print()
            print('Processing KS gamma matrices for target {}: '.format(t), end='')
            sys.stdout.flush()
            reftable_filename = folder + '/ks-gamma-{}.mtx'.format(t)
            difftable_filename = folder + '/ks-gamma-{}-{}.mtx'.format(t, dt)

            reftable = scipy.io.mmread(reftable_filename).toarray()
            print('LR ', end='')
            sys.stdout.flush()

            difftable = scipy.io.mmread(difftable_filename).toarray()
            print('LD ', end='')
            sys.stdout.flush()

            diff_dr = abs(difftable - reftable)
            diff_dr_rel = numpy.divide(diff_dr, difftable)
            print('DDR ', end='')
            sys.stdout.flush()

            maxdiff = numpy.max(diff_dr)
            position = numpy.unravel_index(numpy.argmax(diff_dr), diff_dr.shape)
            refvalue = reftable[position]
            diffvalue = difftable[position]
            maxdiff_rel = maxdiff / diffvalue
            print('MAX ', end='')
            sys.stdout.flush()

            print('\r' + ' '*80 + '\r', end='')
            print('Gamma table {}:'.format(t))
            print('Max element-wise difference: {}'.format(maxdiff))
            if maxdiff > 0:
                print('Max element-wise difference position: {}'.format(position))
                print('Max element-wise difference values (ref, diff): ({}, {})'
                        .format(refvalue, diffvalue))
            print('Max element-wise relative difference: {}'.format(maxdiff_rel))
            print('Shape identical: {}, {} vs {}'.format(
                reftable.shape == difftable.shape, reftable.shape,
                difftable.shape))
            print('Types: {} vs {}'.format(reftable.dtype, difftable.dtype))
            print('='*16)
            print()
            sys.stdout.flush()




def map_over_exds_definitions(exds_names, opname, op, print_status=True):
    i = 1
    definitions = list(get_from_definitions(ExperimentalDatasets, exds_names))
    results = []
    for definition in definitions:
        if opname != '' and print_status == True:
            mpprint('{} ExDs {} ({} / {})'.format(opname, definition.name, i, len(definitions)))
        results.append(op(definition))
        i += 1
    return results
