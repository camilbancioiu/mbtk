import analysis

import expsetup


def configure_objects_subparser__summary(subparsers):
    subparser = subparsers.add_parser('summary')
    subparser.add_argument('tags', type=str, default=None, nargs='?')
    subparser.add_argument('-t', '--target-condset-histogram', action='store_true', default=False)
    subparser.add_argument('-T', '--total-condset-histogram', action='store_true', default=False)
    subparser.add_argument('-m', '--target-mb-analysis', action='store_true', default=False)
    subparser.add_argument('-c', '--target-citr-analysis', action='store_true', default=False)
    subparser.add_argument('--tabular', action='store_true', default=False)
    subparser.set_defaults(function=command_summary)



def configure_objects_subparser__sources(subparsers, experimental_setup):
    subparser_bn = subparsers.add_parser('bn')
    subparser_bn.add_argument('name', type=str, default=None,
                              choices=expsetup.BAYESIAN_NETWORKS)
    subparser_bn.set_defaults(source_type='bn')

    subparser_ds = subparsers.add_parser('ds')
    subparser_ds.add_argument('name', type=str, default=None,
                              choices=expsetup.BAYESIAN_NETWORKS)
    subparser_ds.set_defaults(source_type='ds')



def command_summary(experimental_setup):
    summary = create_summary(experimental_setup)

    if experimental_setup.arguments.tabular:
        print(render_tabular_summary(experimental_setup, summary))
    else:
        print()
        summary_header = dict()
        summary_header.update(experimental_setup.__dict__)
        print('Summary {source_type}'
              ' {bayesian_network_name} {sample_count_string}'
              ' {algorithm_name}:'.format(**summary_header))

        for key, value in summary.items():
            print('\t' + key + ':', value)



def create_summary(experimental_setup):
    summary = dict()
    summary['Runs'] = len(experimental_setup.algorithm_run_parameters)

    citr_analysis = analysis.create_citr_analysis(experimental_setup)
    summary.update(citr_analysis)

    return summary



def render_tabular_summary(experimental_setup, analysis):
    network = experimental_setup.bayesian_network_name.upper()
    samples = experimental_setup.sample_count
    tests = analysis['Total CI count']
    accurate_tests = analysis['Total accurate CI count (%)']
    avg_condset_size = analysis['Avg. cond. set size']
    distance = analysis['Distance']

    accurate_tests = str(accurate_tests).replace('%', '\\%')
    avg_condset_size = f'{avg_condset_size:>5.2f}'

    values = [network, samples, tests, accurate_tests, avg_condset_size, distance]
    values = map(str, values)
    values = '\t& '.join(values)
    row = values + ' \\\\'

    return row
