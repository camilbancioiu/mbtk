
from experimental_dataset import *
from experimental_pipeline import *
import dataset_rcv1v2 as rcv1v2
import utilities as util
import collections
import functools as F
import operator
import itertools


FULL_USE = 1
COMPARE_ONLY = 2
DISABLED = 4

## Helper functions to manipulate collections of definitions

def add_to_definitions(definitions_dict, definitions):
    for definition in definitions:
        definitions_dict[definition.name] = definition

def get_from_definitions(definitions_dict, definition_names):
    if definition_names == ['all']:
        definition_names = sorted(definitions_dict.keys())
    # use dict.get() to return None, instead 
    return map(definitions_dict.get, definition_names)

def get_definition_names_by_tag(definitions_dict, tag):
    definition_names = []
    for definition in definitions_dict.values():
        if tag in definition.tags:
            definition_names.append(definition.name)
    return definition_names


## Utilities for mass ExDs definition

def create_exds_definition_from_industry(industry):
    exds_name = 'industry_{}'.format(industry)
    return ExperimentalDatasetDefinition(exds_name, industry, 0.30, (0.1, 0.9))

def create_exds_definitions_from_industry_list(industries, listID=None):
    definitions = list(map(create_exds_definition_from_industry, industries))
    if listID != None:
        for definition in definitions:
            definition.tags.append(listID)
    return definitions

def create_exds_definitions_from_industry_list_file(listID):
    try:
        industries = util.load_list_from_file('industry_list', listID)
        definitions = create_exds_definitions_from_industry_list(industries, listID)
        return definitions
    except FileNotFoundError:
        print('Industry list not found: {}'.format(listID))
        return []

def create_exds_definitions_from_industry_list_files(listIDs):
    definition_lists = []
    for listID in listIDs:
        definitions = create_exds_definitions_from_industry_list_file(listID)
        definition_lists.append(definitions)
    return itertools.chain(*definition_lists)

