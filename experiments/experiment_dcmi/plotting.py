import itertools
import operator


def make_plot_data(metric, citr):
    data = dict()

    if metric == 'duration':
        for key, results in citr.items():
            data[key] = map(operator.attrgetter('duration'), results)
    if metric == 'duration-cummulative':
        for key, results in citr.items():
            durations = map(operator.attrgetter('duration'), results)
            data[key] = itertools.accumulate(durations)

    return data



def make_plot_Xaxis(data):
    longest_list_of_results = 0
    for results in data.values():
        if len(results) > longest_list_of_results:
            longest_list_of_results = len(results)

    return list(range(longest_list_of_results))



def plot(data, adtree_analysis, plot_save_filename):
    import matplotlib.pyplot as Plotter

    Plotter.figure(figsize=(10, 6))
    # Plotter.clf()
    # Plotter.cla()
    Plotter.rcParams.update({'font.size': 20})
    # pos = Plotter.gca().get_position()
    Plotter.gca().tick_params(axis='both', which='major', pad=8)
    Plotter.gca().margins(0.01)
    Plotter.xlabel('CI Test number')
    Plotter.ylabel('Time (s)')

    Xaxis = None
    for run in sorted(data.keys()):
        Yvalues = list(data[run])
        if Xaxis is None:
            Xaxis = list(range(len(Yvalues)))
        if len(Yvalues) > 0:
            Plotter.plot(Xaxis, Yvalues, lw=1.5)

    legend = make_plot_legend(data, adtree_analysis)
    Plotter.legend(legend)
    Plotter.yscale('log')
    Plotter.grid(True)
    Plotter.title('CI test times')
    Plotter.tight_layout()
    if plot_save_filename is None:
        Plotter.show()
    else:
        Plotter.savefig(plot_save_filename)
        print('Plot saved to {}'.format(plot_save_filename))



def make_plot_legend(data, adtree_analysis):
    legend = list()
    for tag in sorted(data.keys()):
        try:
            analysis = adtree_analysis[tag]
            entry = make_plot_legend_entry_for_adtree_run(tag, analysis)
        except (KeyError, TypeError):
            entry = tag
        legend.append(entry)

    return legend



def make_plot_legend_entry_for_adtree_run(tag, analysis):
    from humanize import naturalsize, naturaldelta
    from datetime import timedelta
    entry = '{} ({size}, {nodes} nodes, {duration} to build)'.format(
        tag,
        size=naturalsize(analysis['size']),
        nodes=analysis['nodes'],
        duration=naturaldelta(timedelta(seconds=analysis['duration'])))
    return entry
