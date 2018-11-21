import itertools
import functools as F
import utilities as util
import operator as op
from pathlib import Path
import sys
from pprint import pformat, pprint
import csv
from definitions import Experiments, ExperimentalDatasets, get_from_definitions
from experimental_pipeline import ExperimentDefinition, Experiment, AlgorithmRunSample
import pickle
import datetime
import numpy

import matplotlib.pyplot as Plotter
import plotting
import sampling

arguments = None


### Analysis-exprun operation 'plot-ksic-stats'
def op_plot_ksic_stats(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Plot KSIC stats', op_plot_ksic_stats_single)

def op_plot_ksic_stats_single(definition):
    stats = pickle.load(open(definition.folder + '/ks/ks_ic_stats.pickle', 'rb'))
    plotpath = definition.folder + '/plots'
    path = Path('./' + plotpath)
    path.mkdir(parents=True, exist_ok=True)
    plotting.plot_ksic_stats(stats, plotpath)
     

### Analysis-exprun operation 'plot-ksic-stats'
def op_print_ksic_stats(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Print KSIC stats', op_print_ksic_stats_single)

def op_print_ksic_stats_single(definition):
    stats = pickle.load(open(definition.folder + '/ks/ks_ic_stats.pickle', 'rb'))
    for Tj in util.list_iteration_keys_Tj(stats.keys()):
        for K in util.list_iteration_keys_K(stats.keys()):
            print('--------------------------------')
            print('KSIC stats for run Tj={} and K={}:'.format(Tj, K))
            keys = util.list_iteration_keys(stats.keys(), Tj, K)
            hits = ([stats[key][0] for key in keys])
            misses = ([stats[key][1] for key in keys])
            avg_hit_rate = sum(hits) * 1.0 / (sum(hits) + sum(misses))
            print('Avg. hit rate: {}'.format(avg_hit_rate))



### Analysis-exprun operation 'plot-iteration-time'
#   The chosen experiment dictates the iteration keys used in plotting
#   create a plot for each combination Tj x K, containing
#     - the reference plot of iteration times from the ExDs KSFDB
#     - the plot of iteration times from the ExpRun KSFDB
def op_plot_iteration_time(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Plot iteration times', op_plot_iteration_time_single)

def op_plot_iteration_time_single(definition):
    plotpath = definition.folder + '/plots'
    path = Path('./' + plotpath)
    path.mkdir(parents=True, exist_ok=True)
    exprun_fdb = pickle.load(open(definition.folder + '/ks/ks_compare_fdb.pickle', 'rb'))
    exds_fdb = pickle.load(open(definition.exds_definition.folder + '/ks/ks_fdb.pickle', 'rb'))
    targets = util.list_iteration_keys_Tj(exprun_fdb.keys())
    K_values = util.list_iteration_keys_K(exprun_fdb.keys())
    for Tj in util.list_iteration_keys_Tj(exprun_fdb.keys()):
        for K in util.list_iteration_keys_K(exprun_fdb.keys()):
            iteration_keys = util.list_iteration_keys(exprun_fdb.keys(), Tj, K)
            iteration_numbers = [key[1] for key in iteration_keys]
            exprun_times = [exprun_fdb[key][1]['duration'] for key in iteration_keys]
            exds_times = [exds_fdb[key][1]['duration'] for key in iteration_keys]
            Plotter.figure(figsize=(10,6))
            #Plotter.clf()
            #Plotter.cla()
            Plotter.rcParams.update({'font.size': 20})
            pos = Plotter.gca().get_position()
            Plotter.gca().tick_params(axis='both', which='major', pad=8)
            Plotter.gca().margins(0.01)
            Plotter.xlabel('Iteration number')
            Plotter.ylabel('Time (s)')
            Plotter.plot(iteration_numbers, exds_times, lw=1.5)
            Plotter.plot(iteration_numbers, exprun_times, lw=1.5)
            axisYConfig = definition.config['plots']['iteration_time']['axisY']
            Plotter.axis([-10, len(iteration_numbers) - 1, axisYConfig[0], axisYConfig[1]])
            (xticks, ticklabels) = Plotter.xticks()
            xticks[-1] = iteration_numbers[-1]
            xticks[-2] -= 200
            xticks = xticks[1:]
            Plotter.xticks(xticks)
            (xticks, ticklabels) = Plotter.xticks()
            xticks[0] = -10
            ticklabels = list(map(str, map(int, xticks)))
            ticklabels[0] = '0'
            Plotter.xticks(xticks, ticklabels)
            Plotter.legend(definition.config['plots']['iteration_time']['legend'])
            Plotter.title(definition.config['plots']['iteration_time']['title'])
            Plotter.tight_layout()
            Plotter.savefig(plotpath + '/ks_iteration_durations__{}__T{}_K{}.png'.format(definition.name, Tj, K))


### Analysis-exprun operation 'print-algrun-durations'
def op_print_algrun_durations(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Print AlgorithmRun durations', op_print_algrun_durations_single)

def op_print_algrun_durations_single(definition):
    experiment = Experiment(definition)
    algorithm_runs = experiment.load_saved_runs()
    total_duration = datetime.timedelta(milliseconds=0)
    for run in algorithm_runs: 
        duration = datetime.timedelta(milliseconds=run.duration)
        print('AlgorithmRun {} duration: {}'.format(run.name, duration))
        total_duration += duration
    print('Total: {}'.format(total_duration))


### Analysis-exprun operation 'print-algrun-results'
def op_print_algrun_results(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Print AlgorithmRun results', op_print_algrun_results_single)

def op_print_algrun_results_single(definition):
    experiment = Experiment(definition)
    algorithm_runs = experiment.load_saved_runs()
    for run in algorithm_runs: 
        print('AlgorithmRun {} results:'.format(run.name))
        print(sorted(run.selected_features))


### Analysis-exprun operation 'generate-algrun-samples'
def op_generate_algrun_samples(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Generate AlgorithmRun samples', op_generate_algrun_samples_single)

def op_generate_algrun_samples_single(definition):
    Experiment(definition).generate_algrun_samples()


### Analysis-exprun operation 'print-accuracy-stats'
def op_print_accuracy_stats(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Accuracy stats', op_print_accuracy_stats_single)

def op_print_accuracy_stats_single(definition):
    experiment = Experiment(definition)
    samples = sampling.KS_IGt_Samples(experiment)

    if arguments.custom_q == True:
        ## The optional argument --custom-q ensures fair comparison with
        ## the results in the C2FSA article, which didn't evaluate over all
        ## possible Q values, but had a limited range instead.
        custom_Q_values = list(range(1, 300))
        custom_Q_values.extend(list(range(300, 2000, 50))[1:])
        samples = samples.filterCustom(lambda s: s.Q in custom_Q_values).new()
        Q_values = custom_Q_values
    else:
        Q_values = sorted(list(samples.values("Q")))

    algorithms = samples.values("algorithm")
    targets = samples.values("target")
    for target in targets:
        for algorithm in algorithms:
            selected_samples = samples.reset().filter("target", target).filter("algorithm", algorithm)
            if algorithm != 'KS':
                max_accuracy = selected_samples.max("accuracy")
                max_accuracy_at_Q = selected_samples.argmax("Q", "accuracy")
                avg_accuracy = selected_samples.average("accuracy")
                print("Accuracy for \tT{} {}: \t\tmax {:.4f} @ Q={}, \taverage {:.4f}".format(
                    target, algorithm, max_accuracy, max_accuracy_at_Q, avg_accuracy))
            if algorithm == 'KS':
                selected_samples = selected_samples.new()
                K_values = selected_samples.values("K")
                for K in K_values:
                    selected_samples = selected_samples.reset().filter("K", K)
                    max_accuracy = selected_samples.max("accuracy")
                    max_accuracy_at_Q = selected_samples.argmax("Q", "accuracy")
                    avg_accuracy = selected_samples.average("accuracy")
                    print("Accuracy for \tT{} {} K{}: \tmax {:.4f} @ Q={}, \taverage {:.4f}".format(
                        target, algorithm, K, max_accuracy, max_accuracy_at_Q, avg_accuracy))


### Analysis-exprun operation 'custom-fix'
def op_custom_fix():
    definition = Experiments['ks_0']
    experiment = Experiment(definition)
    with open('{}/samples/algrun-samples-1.pickle'.format(definition.folder), 'rb') as f:
        samples_1 = pickle.load(f)
    with open('{}/samples/algrun-samples-2.pickle'.format(definition.folder), 'rb') as f:
        samples_2 = pickle.load(f)

    collection_1 = sampling.KS_IGt_Samples(experiment, False)
    collection_1.original_samples = samples_1
    collection_1.reset()

    collection_2 = sampling.KS_IGt_Samples(experiment, False)
    collection_2.original_samples = samples_2
    collection_2.reset()

    print(collection_1.values("K"))
    print(collection_2.values("K"))

    print(len(collection_1))
    print(len(collection_2))

    common_K_values = collection_1.values("K").intersection(collection_2.values("K"))
    print(common_K_values)

    fixed_samples = list(filter(lambda sample: (not sample.K in common_K_values), samples_1))
    fixed_samples.extend(samples_2)
    fixed_collection = sampling.KS_IGt_Samples(experiment, False)
    fixed_collection.original_samples = fixed_samples
    fixed_collection.reset()

    print(fixed_collection.values("K"))
    print(len(fixed_collection))

    with open('{}/samples/algrun-samples.pickle'.format(definition.folder), 'wb') as f:
        pickle.dump(fixed_samples, f)

    print('Done')

### Analysis-exprun operation 'plot-accuracy-igt-vs-ks'
### This operation creates plots whic compare the classification accuracy
### of features selected by IGt versus those selected by KS. Only the first
### two experiments provided as argument is used. The first experiment provides 
### the IGt samples and the second provides the KS samples.
def op_plot_accuracy_igt_vs_ks(experiment_names):
    igt_definition = Experiments[experiment_names[0]]
    igt_experiment = Experiment(igt_definition)
    igt_samples = sampling.KS_IGt_Samples(igt_experiment).filter("algorithm", "IG").new()

    ks_definition = Experiments[experiment_names[1]]
    ks_experiment = Experiment(ks_definition)
    ks_samples = sampling.KS_IGt_Samples(ks_experiment).filter("algorithm", "KS").new()

    Plotter.figure(figsize=(8,7))
    Plotter.clf()
    Plotter.cla()
    pos = Plotter.gca().get_position()
    Plotter.rcParams.update({'font.size': 14})
    Plotter.gca().tick_params(axis='both', which='major', pad=18)
    Plotter.xlabel('Q (selected features)')
    Plotter.ylabel('Classifier accuracy')

    targets = ks_samples.reset().values("target")
    K_values = ks_samples.reset().values("K")
    Q_values = sorted(list(ks_samples.reset().values("Q")))
    Q_values = Q_values[0:200]
    print("Values: K={}, target={}".format(K_values, targets))
    print("MaxQ: {}".format(max(Q_values)))
    igt_accuracies = igt_samples.reset().filter("target", 0).sort("Q").property("accuracy")
    igt_accuracies = igt_accuracies[0:200]
    for target in targets:
        ks_accuracies = ks_samples.reset().filter("target", target).sort("Q").new()
        for K in K_values:
            print("Plotting accuracy between IGt ({}) and KS ({}) for T{} and K{}...".format(
                igt_definition.name, ks_definition.name, target, K))
            ks_accuracies_list = ks_accuracies.reset().filter("K", K).property("accuracy")
            ks_accuracies_list = ks_accuracies_list[0:200]
            Plotter.clf()
            Plotter.cla()
            Plotter.plot(Q_values, ks_accuracies_list, 'b', lw=3)
            Plotter.plot(Q_values, igt_accuracies, 'g', lw=1)
            Plotter.title('K = {}'.format(K))
            Plotter.legend(['KS', 'IG'])
            Plotter.xlabel('Q (selected features)')
            Plotter.ylabel('Classifier accuracy')
            Plotter.axis([min(Q_values), 175, 0.65, 1])
            Plotter.gca().set_autoscale_on(False)
            Plotter.tight_layout()
            ks_experiment.ensure_subfolder('plots')
            filename = '{}/plots/accuracy_igt_vs_ks_T{}_K{}.png'.format(ks_definition.folder, target, K)
            Plotter.savefig(filename)
            print("Plot saved to {}.".format(filename))


### Helper functions

def map_over_experiment_definitions(experiment_names, opname, op, print_status=True):
    i = 1
    definitions = list(get_from_definitions(Experiments, experiment_names))
    results = []
    for definition in definitions:
        if opname != '' and print_status == True:
            print('\n{}: Experiment {} ({} / {})'.format(opname, definition.name, i, len(definitions)))
        results.append(op(definition))
        i += 1
    return results
