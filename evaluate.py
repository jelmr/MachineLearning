import csv
import argparse
import crps
import numpy as np

def calculate_score(results, input):
    results_reader = csv.reader(results, delimiter=',')
    input_reader = csv.reader(input, delimiter=',')

    results_header = results_reader.next()
    input_header = input_reader.next()

    expected_id = input_header.index('Expected')

    predictions = np.array([row[1:] for row in results_reader]).astype(np.float)
    actuals = np.array([row[expected_id] for row in input_reader]).astype(np.float)

    crps_score = crps.calc_crps(predictions, actuals)

    print crps_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=argparse.FileType('r'),
                        help=("File containing the results of the algorithm."))
    parser.add_argument('input', type=argparse.FileType('r'),
                        help=("Path to the original input file used to generate the results."))
    args = parser.parse_args()

    calculate_score(args.results, args.input)

