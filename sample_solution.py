"""
An example script which takes a train/test file as input and
produces a solution file.

Usage:
    python sample_solution.py --input ./test_2014.csv --output sampleSolution.csv

@author Alex Kleeman, The Climate Corporation
"""
import csv
import sys
import logging
import argparse
import numpy as np

# configure logging
logger = logging.getLogger("example")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def sigmoid(center, length):
    # http://en.wikipedia.org/wiki/Sigmoid_function
    xs = np.arange(length)
    return 1. / (1 + np.exp(-(xs - center)))


def produce_solution(args):
    # wrap the inputs and outputs in csv interpreters
    writer = csv.writer(args.output, delimiter=',')
    reader = csv.reader(args.input, delimiter=',')
    # read in the header
    header = reader.next()
    # the solution header is an Id, then 70 discrete cumulative probabilities
    solution_header = ['Id']
    # Add the fields defining the cumulative probabilites
    # ex: Predicted5 is interpreted as the probability that the hour's rainfall
    #     accumulation was less than or equal to 5mm.
    solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, 70)])
    # write the header to file
    writer.writerow(solution_header)
    # determine which columns hold Id, Rainfall Rate and Time
    id_ind = header.index('Id')
    # the rr in rr1 stands for RainfallRate
    rr1_ind = header.index('RR1')
    time_ind = header.index('TimeToEnd')
    # iterate over each row and estimate a rainfall probability distribution
    for i, row in enumerate(reader):
        # Each row in the file represents a single hour's observations at some
        # location and consists of a set of comma delimited fields.  Often
        # there is more than one radar observation per hour, so some fields
        # have multiple values per row, each corresponding of a single radar
        # volume scan.  For example, if five scans were completed within a
        # single hour the reflectivity field may look something like:
        #
        #     "17.5 30.5 42.0 38.5 37".
        #
        # Similarly, in this next line, we find the rainfall rate field ('RR1')
        # and then convert it from a space delimited string to a float array.
        rr1 = np.array(row[rr1_ind].split(' '), dtype='float')
        # compute the mean of the sample (units at this point are mm/hr)
        avg_rr1 = np.mean(rr1)
        # extract the times to end
        times = np.array(row[time_ind].split(' '), dtype='float')
        time_period = (np.max(times) - np.min(times) + 6.) / 60.
        # approximate the distribution of hourly totals
        approx_rr1 = sigmoid(avg_rr1 * time_period, 70)
        # determine the current Id
        id_num = row[id_ind]
        # write the solution row
        solution_row = [id_num]
        solution_row.extend(approx_rr1)
        writer.writerow(solution_row)
        # Every 1000 rows send an update to the user for progress tracking.
        if i % 1000 == 0:
            logger.info("Completed row %d" % i)

if __name__ == "__main__":
    # set up logger
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=argparse.FileType('r'),
                        help=("path to an input file, this will "
                              "typically be train_2013.csv or "
                              "test_2014.csv"))
    parser.add_argument('--output', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help=("path to an output file, "
                              "defaults to stdout"))
    # parse the arguments and run the handler associated with each task
    args = parser.parse_args()
    produce_solution(args)
