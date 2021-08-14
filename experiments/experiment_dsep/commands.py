import pickle
import analysis

import expsetup


def configure_objects_subparser__summary(subparsers):
    subparser = subparsers.add_parser('summary')
    subparser.add_argument('tags', type=str, default=None, nargs='?')
    subparser.add_argument('--refresh', action='store_true', default=False)
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
    summaries = experimental_setup.paths.Summaries

    summary_path = summaries / 'summary.pickle'
    summary = None
    cached = ''
    try:
        if experimental_setup.arguments.refresh:
            raise FileNotFoundError
        with summary_path.open('rb') as f:
            summary = pickle.load(f)
        cached = ' (cached)'
    except FileNotFoundError:
        summary = create_summary(experimental_setup)
        cached = ''
        with summary_path.open('wb') as f:
            pickle.dump(summary, f)

    print()
    print('Summary{}:'.format(cached))

    for key, value in summary.items():
        print('\t' + key, value)



def create_summary(experimental_setup):
    summary = dict()
    summary['Runs:'] = len(experimental_setup.algorithm_run_parameters)

    citr_analysis = analysis.create_citr_analysis(experimental_setup)

    summary.update(citr_analysis)

    return summary
