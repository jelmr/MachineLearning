import csv
import argparse
import crps
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import normal

def calculate_score(args):
    results_reader = csv.reader(args.results, delimiter=',')
    input_reader = csv.reader(args.input, delimiter=',')

    results_header = results_reader.next()
    input_header = input_reader.next()

    expected_id = input_header.index('Expected')

    thresholds = np.arange(70)
    predictions = np.array([row[1:] for row in results_reader]).astype(np.float)
    actuals = np.array([row[expected_id] for row in input_reader]).astype(np.float)

    crps_score = crps.calc_crps(thresholds, predictions, actuals)

    plot_graph(thresholds, predictions, actuals, args.plot)

    print crps_score


def plot_graph(thresholds, predictions, actuals, name):
    vals = []
    for row, actual in zip(predictions, actuals):
        sum = 0
        for idx, x in enumerate(row):
            if idx < actual:
                sum += x
            else:
                sum += (1. - x)
        vals.append(sum)


    plt.hist(vals, bins=140)
    plt.title("Badness frequency plot")
    plt.xlabel("Badness")
    plt.ylabel("Frequency")
    plt.savefig(name, format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=argparse.FileType('r'),
                        help=("File containing the results of the algorithm."))
    parser.add_argument('input', type=argparse.FileType('r'),
                        help=("Path to the original input file used to generate the results."))
    parser.add_argument('--plot', default="graph.png", type=argparse.FileType('w'),
                        help=("Name for the badness plot."))
    args = parser.parse_args()

    calculate_score(args)

