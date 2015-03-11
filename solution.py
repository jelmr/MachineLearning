import csv
import sys
import logging
import argparse
import numpy as np
from functools import partial
import scipy
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, metrics

NUMBER_OF_PREDICTIONS = 70
NUMBER_OF_FEATURES = 19
OUTLIERS_VALUE = 9999

logger = logging.getLogger("example")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)




def produce_solution(args):
    writer = csv.writer(args.output, delimiter=',')
    reader_train = csv.reader(args.train, delimiter=',')
    reader_test = csv.reader(args.test, delimiter=',')

    write_header(writer, NUMBER_OF_PREDICTIONS)

    train_data, expected_train_values = preprocess_data(*split_dataset(np.array(list(reader_train))), remove_high_rr=True)
    test_data, expected_test_values = preprocess_data(*split_dataset(np.array(list(reader_test))), remove_high_rr=False)

    predictor = {
            'rr1': partial(train_rr1, train_data),
            'svr': partial(train_svr, train_data, expected_train_values),
            'nn': partial(train_nn , train_data, expected_train_values)
            }[args.method]

    print_solution_distribution(predictor(), test_data, writer)

def split_dataset(data):
    expected_id = np.where(data[0] == 'Expected')
    expected_values = data[1:,expected_id].flatten()

    # Remove the expected values column from the data.
    data = np.delete(data, expected_id, 1)
    return data, expected_values

def write_header(writer, n):
    solution_header = ['Id']
    solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, n)])
    writer.writerow(solution_header)

def preprocess_data(data, expected_values, remove_high_rr=False):
    logger.info("Starting preprocessing.")
    header = data[0]
    # Flatten data by taking mean of measurements.
    # TODO: Use scipy/numpy methods?
    data = map(lambda y:
            map(lambda x: np.mean(np.array(x.split(' ')).astype(np.float)), y),
            data[1:])

    # Correct RR3
    for x in data:
        rr3_id = np.where(header == 'RR3')[0][0]
        x[rr3_id] = abs(x[rr3_id])

    # Replace nan by 0s
    data = np.nan_to_num(data)

    # Remove entries where Expected < 70
    if remove_high_rr:
        a = map(lambda x: float(x) < 70, expected_values)
        b = filter(lambda x: a[x], range(len(expected_values)))
        data = data[b]
        expected_values = filter(lambda x: float(x)<70, expected_values)


    # TODO: Use scipy/numpy methods?
    # Remove extreme values (measurement errors)
    data = np.array(map(lambda y:
            map(lambda x: 0 if x > OUTLIERS_VALUE or x < -OUTLIERS_VALUE else x, y), data))

    logger.info("Done with preprocessing.")

    return np.vstack((header, data)), expected_values


def print_solution_distribution(predict, data, writer):

    for i, row in enumerate(data[1:]):
        prediction = predict(row)

        solution_row = [row[np.where(data[0] == 'Id')][0][0]]
        solution_row.extend(prediction)
        writer.writerow(solution_row)

        if i % 1000 == 0:
            logger.info("Completed row %d" % i)

def wrap_threshold_distribtuion(prediction):
    return [0 if x < prediction else 1 for x in range(70)]

def wrap_sigmoid_distribution(prediction, length=70):
    xs = np.arange(length)
    return 1. / (1 + np.exp(-(xs - prediction)))

def train_rr1(data):
    return lambda x: wrap_sigmoid_distribution(np.float(x[np.where(data[0]=='RR1')[0][0]]))


def train_svr(data, expected_values):
    logger.info("Starting feature reduction.")
    data = reduce_features(data[1:], NUMBER_OF_FEATURES)
    logger.info("Done with feature reduction.")

    logger.info("Starting SVR training.")
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(data, expected_values)
    logger.info("Done with SVR training.")
    return lambda x: wrap_threshold_distribtuion(clf.predict(x))

def train_nn(data, expected_values):
    logger.info("Starting feature reduction.")
    X = np.asarray(data[1:], 'float64')
    logger.info("Done with feature reduction.")
    Y = expected_values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)

    logger.info("Starting NeuralNetwork training.")

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 1.0


    clf.fit(X_train, Y_train)


    # Evaluation
    #TODO: Make unified evaluation
    logger.info("Logistic regression using RBM features:\n%s\n" % (
            metrics.classification_report(
            Y_test,
            clf.predict(X_test))))

    logger.info("Done with NeuralNetwork training.")
    return lambda x: wrap_threshold_distribtuion(np.array(clf.predict(x)).astype(float))


def reduce_features(data, number_of_features):
    pca = PCA(n_components=number_of_features)
    data = pca.fit_transform(data)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=argparse.FileType('r'),
                        help=("The trainig data."))
    parser.add_argument('test', type=argparse.FileType('r'),
                        help=("The test data."))
    parser.add_argument('method', choices=['rr1', 'svr', 'nn'],
                        help=("Method to be used to generate the solution."))
    parser.add_argument('--output', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help=("path to an output file, "
                              "defaults to stdout"))

    args = parser.parse_args()
    produce_solution(args)
