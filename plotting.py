import matplotlib.pyplot as Plotter
import utilities as util

def plot_ksic_stats(stats, folder):
    for Tj in util.list_iteration_keys_Tj(stats.keys()):
        for K in util.list_iteration_keys_K(stats.keys()):
            plot_ksic_stats_per_run(stats, folder, Tj, K)

def plot_ksic_stats_per_run(stats, folder, Tj, K):
    selected_keys = util.list_iteration_keys(stats.keys(), Tj, K)
    iteration_numbers = [key[1] for key in selected_keys]
    hit_rates = []
    for key in selected_keys:
        (hits, misses) = stats[key]
        hit_rates.append((1.0 * hits) / (1.0 * hits + misses))
    reset_plotter()
    Plotter.plot(iteration_numbers, hit_rates)
    tickstep = 400
    xticks = [iteration_numbers[i] for i in range(0, len(iteration_numbers), tickstep)]
    xticks.append(iteration_numbers[-1:][0])
    
    Plotter.xticks(xticks, xticks)
    Plotter.show()


def reset_plotter():
    #Plotter.clf()
    #Plotter.cla()
    Plotter.rcParams.update({'font.size': 10})
    Plotter.gca().tick_params(axis='both', which='major', pad=18)
    Plotter.figure(figsize=(12,6))


