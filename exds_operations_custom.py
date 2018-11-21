import definitions as Definitions
import dataset_rcv1v2 as rcv1v2
from experimental_dataset_stats import ExperimentalDatasetStats, ExperimentalDatasetStatsError
import utilities as util

def op_custom(exds_names, arguments):
    op = ''
    if arguments.targets[0] == 'create_industry_list_lims':
        create_industry_list_with_limited_docs(exds_names, arguments)
    elif arguments.targets[0] == 'filter_industries_by_voc_size':
        filter_industries_by_voc_size(exds_names, arguments)


def create_industry_list_with_limited_docs(exds_names, arguments):
    listID = arguments.targets[1]
    mindocs = int(arguments.targets[2])
    maxdocs = int(arguments.targets[3])
    print('Creating industry list file {} with docs between {} and {}.'.format(listID, mindocs, maxdocs))
    industries = rcv1v2.create_industry_list_between(mindocs, maxdocs)
    util.save_list_to_file(industries, 'industry_list', listID)

def filter_industries_by_voc_size(exds_names, arguments):
    print(arguments)
    listID = arguments.targets[1]
    minvoc = int(arguments.targets[2])
    maxvoc = int(arguments.targets[3])
    saveListID = arguments.targets[4]
    print('Filtering industry list file {} with vocabulary size between {} and {}.'.format(listID, minvoc, maxvoc))
    industries = util.load_list_from_file('industry_list', listID)

    filtered_industries = []
    for industry in industries:
        definition = Definitions.create_exds_definition_from_industry(industry)
        try:
            stats = ExperimentalDatasetStats(definition)
            if minvoc <= stats['full_vocab_size'] <= maxvoc:
                filtered_industries.append(industry)
                print('Accepting {} with vocabulary size {}.'.format(industry, stats['full_vocab_size']))
            else:
                print('Rejecting {} with vocabulary size {}.'.format(industry, stats['full_vocab_size']))
        except ExperimentalDatasetStatsError:
            print('Cannot load stats of ExDs {}. Build its stats first.'.format(definition.name))

    util.save_list_to_file(filtered_industries, 'industry_list', saveListID)
    print('Filtered industries saved: {}\n{}'.format(len(filtered_industries), filtered_industries))

