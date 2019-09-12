import itertools


def make_plot_data(plot_what, citr):
    data = dict()

    if plot_what == 'duration':
        for key, results in citr.items():
            data[key] = [result.duration for result in results]
    if plot_what == 'duration-cummulative':
        for key, results in citr.items():
            durations = [result.duration for result in results]
            durations_cummulative = list(itertools.accumulate(durations))
            data[key] = durations_cummulative

    return data



def make_plot_Xaxis(data):
    longest_list_of_results = 0
    for results in data.values():
        if len(results) > longest_list_of_results:
            longest_list_of_results = len(results)

    return list(range(longest_list_of_results))



def plot(data, adtree_analysis, plot_save_filename):
    import matplotlib.pyplot as Plotter

    Xaxis = make_plot_Xaxis(data)

    Plotter.figure(figsize=(10, 6))
    # Plotter.clf()
    # Plotter.cla()
    Plotter.rcParams.update({'font.size': 20})
    # pos = Plotter.gca().get_position()
    Plotter.gca().tick_params(axis='both', which='major', pad=8)
    Plotter.gca().margins(0.01)
    Plotter.xlabel('CI Test number')
    Plotter.ylabel('Time (s)')

    for run in sorted(data.keys()):
        Yvalues = data[run]
        if len(Yvalues) > 0:
            Plotter.plot(Xaxis, Yvalues, lw=1.5)

    legend = make_plot_legend(data, adtree_analysis)
    Plotter.legend(legend)
    # Plotter.yscale('log')
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
    for run in sorted(data.keys()):
        if run in adtree_analysis:
            analysis = adtree_analysis[run]
            entry = make_plot_legend_entry_for_adtree_run(run, analysis)
        else:
            entry = run
        legend.append(entry)

    return legend



def make_plot_legend_entry_for_adtree_run(run, analysis):
    from humanize import naturalsize, naturaldelta
    from datetime import timedelta
    entry = '{} ({size}, {nodes} nodes, {duration} to build)'.format(
        run,
        size=naturalsize(analysis['size']),
        nodes=analysis['nodes'],
        duration=naturaldelta(timedelta(seconds=analysis['duration'])))
    return entry
