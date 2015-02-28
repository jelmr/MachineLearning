import csv
import sys
import logging
import argparse
import numpy as np
from sklearn.decomposition import PCA

NUMBER_OF_PREDICTIONS = 70
NUMBER_OF_FEATURES = 5
OUTLIERS_VALUE = 9999

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

    write_header(writer, NUMBER_OF_PREDICTIONS)

    header = reader.next()
    id_ind = header.index('Id')
    rr1_ind = header.index('RR1')
    time_ind = header.index('TimeToEnd')

    data = preprocess_data(list(reader))

    #for i, row in enumerate(reader):



        # rr1 = np.array(row[rr1_ind].split(' '), dtype='float')
        # avg_rr1 = np.mean(rr1)

        # times = np.array(row[time_ind].split(' '), dtype='float')
        # time_period = (np.max(times) - np.min(times) + 6.) / 60.

        # approx_rr1 = sigmoid(avg_rr1 * time_period, 70)

        # id_num = row[id_ind]
        # solution_row = [id_num]
        # solution_row.extend(approx_rr1)
        # writer.writerow(solution_row)

        # if i % 1000 == 0:
        #     logger.info("Completed row %d" % i)

def write_header(writer, n):
    solution_header = ['Id']
    solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, n)])
    writer.writerow(solution_header)

def preprocess_data(data):
    # Flatten data by taking mean of measurements.
    data = map(lambda y:
            map(lambda x: np.mean(np.array(x.split(' ')).astype(np.float)), y),
            data)

    # Replace nan by 0s
    data = np.nan_to_num(data)

    # Remove extreme values (measurement errors)
    data = np.array(map(lambda y:
            map(lambda x: 0 if x > OUTLIERS_VALUE or x < -OUTLIERS_VALUE else x, y), data))


    pca = PCA(n_components=NUMBER_OF_FEATURES)
    data = pca.fit_transform(data)
    print data
    return data

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
