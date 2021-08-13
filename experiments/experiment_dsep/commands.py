import pickle
import analysis


def configure_objects_subparser__summary(subparsers):
    subparser = subparsers.add_parser('summary')
    subparser.add_argument('tags', type=str, default=None, nargs='?')
    subparser.add_argument('--refresh', action='store_true', default=False)
    subparser.set_defaults(function=command_summary)



def configure_objects_subparser__sources(subparsers, experimental_setup):
    subparser_bn = subparsers.add_parser('bn')
    subparser_bn.add_argument('name', type=str, default=None,
                              choices=experimental_setup.BayesianNetworks)
    subparser_bn.set_defaults(source='bn')

    subparser_ds = subparsers.add_parser('ds')
    subparser_ds.add_argument('name', type=str, default=None,
                              choices=experimental_setup.BayesianNetworks)
    subparser_ds.set_defaults(source='ds')



def command_summary(experimental_setup):
    summaries = experimental_setup.Paths.Summaries

    summary_path = summaries / 'summary.pickle'
    summary = None
    cached = ''
    try:
        if experimental_setup.Arguments.refresh:
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
    algruns = list(experimental_setup.AlgorithmRunParameters)
    summary['Runs:'] = len(algruns)

    citr_analysis = analysis.create_citr_analysis(algruns)

    summary.update(citr_analysis)

    return summary
