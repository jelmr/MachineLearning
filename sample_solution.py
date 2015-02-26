import csv
import sys
import logging
import argparse
import numpy as np

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
    writer = csv.writer(args.output, delimiter=',')
    reader = csv.reader(args.input, delimiter=',')

    header = reader.next()

    solution_header = ['Id']
    solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, 70)])
    writer.writerow(solution_header)

    id_ind = header.index('Id')
    rr1_ind = header.index('RR1')
    time_ind = header.index('TimeToEnd')

    for i, row in enumerate(reader):
        rr1 = np.array(row[rr1_ind].split(' '), dtype='float')
        avg_rr1 = np.mean(rr1)

        times = np.array(row[time_ind].split(' '), dtype='float')
        time_period = (np.max(times) - np.min(times) + 6.) / 60.

        approx_rr1 = sigmoid(avg_rr1 * time_period, 70)

        id_num = row[id_ind]
        solution_row = [id_num]
        solution_row.extend(approx_rr1)
        writer.writerow(solution_row)

        if i % 1000 == 0:
            logger.info("Completed row %d" % i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=argparse.FileType('r'),
                        help=("path to an input file, this will "
                              "typically be train_2013.csv or "
                              "test_2014.csv"))
    parser.add_argument('--output', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help=("path to an output file, "
                              "defaults to stdout"))

    args = parser.parse_args()
    produce_solution(args)
