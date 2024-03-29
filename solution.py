import csv
import sys
import logging
import argparse
import numpy as np
import pickle
import math
import gc
from functools import partial
import scipy

from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn import svm
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import SGDClassifier
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

    train_data, expected_train_values = split_dataset(np.array(list(reader_train)))
    if args.evaluate:
        test_data, expected_test_values = split_dataset(np.array(list(reader_test)))
    else:
        test_data = np.array(list(reader_test))
        expected_test_values = None
    test_ids = test_data[:,0]
    train_data = train_data[:,[0,7,8,9,17]]
    test_data = test_data[:,[0,7,8,9,17]]
    gc.collect()
    predictor = {
            'rr1': partial(train_rr1, train_data, expected_train_values),
            'svr': partial(train_svr, train_data, expected_train_values),
            'nn': partial(train_nn , train_data, expected_train_values),
            'lg': partial(train_lg , train_data, expected_train_values),
            'tc': partial(train_threshold_classifier, train_data, expected_train_values, test_data, expected_test_values)
            }[args.method]

    train_data, expected_train_values = None, None
    gc.collect()
    print_solution_distribution(predictor(), test_data, expected_test_values, test_ids, writer)

def train_threshold_classifier(train_data, expected_train_values, test_data, expected_test_values):
    boolean_predictor = train_boolean_predictor(train_data[:,[1,2,3,4]], expected_train_values)

    if expected_test_values is not None:
        evaluate(boolean_predictor, test_data, expected_test_values)

    reg_data, expected_reg_values = zip(*(filter(lambda (row, exp): int(float(exp)) != 0 ,zip(train_data[1:], expected_train_values))))
    print "Reg data: %d\nReg exp: %d\n" % (len(reg_data), len(expected_reg_values))
    reg_predictor = train_lg(np.vstack((train_data[0],reg_data)), expected_reg_values)



    return lambda x: [1]*70 if boolean_predictor(x[:,[1,2,3,4]]) == 0 else reg_predictor(x)

def evaluate(boolean_predictor, test_data, expected_test_values):
    test_data, expected_test_values = preprocess_data(test_data, expected_test_values, remove_high_rr=False)
    test_data = test_data[:,[1,2,3,4]]
    test_data = np.array(test_data[1:]).astype(np.float)
    expected_test_values = map(lambda x: 0 if int(float(x)) == 0 else 1, expected_test_values)
    correct = 0
    incorrect = 0


    logger.info("Boolean classification: \n%s" % (
            metrics.classification_report(
            np.array(expected_test_values).astype(np.float).astype(int),
            boolean_predictor(test_data))))

    for row, expected in zip(test_data, expected_test_values):
        p = boolean_predictor(row)
        if (int(p) == int(float(expected))):
            correct += 1
        else:
            incorrect += 1

    print "Correct: %d (%.2f%%) \nIncorrect: %d (%.2f%%)" % (correct, 100.*correct/(incorrect+correct),incorrect, 100.*incorrect/(incorrect+correct))

def train_boolean_predictor(data, expected_values):

    data, expected_values = preprocess_data(data, expected_values, remove_high_rr=False)
    # Keep only entries where RR1 is close to Expected.
    rr1_id = np.where(data[0] == 'RR1')[0][0]
    data, expected_values = zip(*(filter(lambda (row, exp): abs(float(row[rr1_id])- float(exp)) < 3 ,zip(data[1:], expected_values))))
    print "Data: %d\nExp: %d\n" % (len(data), len(expected_values))
    #data = data[1:]

    expected_values = map(lambda x: 0 if int(float(x)) == 0 else 1, expected_values)

    # Make an equal amount of zeroes and ones.
    if(False):
        ones = filter(lambda (x,y): y == 1, zip(data,  expected_values))
        zeroes = filter(lambda (x,y): y == 0, zip(data,  expected_values))
        m = min(len(ones), len(zeroes))
        data, expected_values = zip(*(ones[:m] + zeroes[:m]))

    #clf = RadiusNeighborsClassifier(radius=2.5)

    #clf = KNeighborsClassifier(n_neighbors=13)

    #clf = linear_model.LogisticRegression()

    #logistic = linear_model.LogisticRegression()
    #rbm = BernoulliRBM(random_state=0, verbose=True)
    #clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    #clf = tree.DecisionTreeClassifier(max_depth=3)

    clf = SGDClassifier(loss="log", penalty="l2", n_jobs=-1, class_weight={0:1, 1:10})

    clf.fit(data, expected_values)
    return clf.predict


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

def preprocess_data(data, expected_values, remove_high_rr=False, delete_features=False):
    logger.info("Starting preprocessing.")
    header = data[0]

        #'Id',  0
        #'TimeToEnd' 1
        #'DistanceToRadar' 2
        #'Composite' 3
        #'HybridScan' 4
        #'HydrometeorType' 5
        #'Kdp' 6
        #'RR1' 7
        #'RR2' 8
        #'RR3' 9
        #'RadarQualityIndex' 10
        #'Reflectivity' 11
        #'ReflectivityQC' 12
        #'RhoHV' 13
        #'Velocity' 14
        #'Zdr' 15
        #'LogWaterVolume' 16
        #'MassWeightedMean' 17
        #'MassWeightedSD' 18

    if delete_features:
        keep = [7,8,9,17]
        #keep = [6,7,8,9,10, 11, 12, 13, 14, 15, 16,17]
        header = header[keep]
        data = data[:,keep]

    # Flatten data by taking mean of measurements.
    data = np.vectorize(lambda x: np.mean(np.array(x.split(' ')).astype(np.float)))(data[1:])

    #data = map(lambda y:
    #    map(lambda x: np.mean(np.array(x.split(' ')).astype(np.float)), y), data[1:])

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


    # Remove extreme values (measurement errors)
    data[np.logical_or(data < -OUTLIERS_VALUE, data > OUTLIERS_VALUE)] = 0

    #data = np.array(map(lambda y:
    #        map(lambda x: math.log(abs(x)+0.00001) , y), data))

    logger.info("Done with preprocessing.")

    return np.vstack((header, data)), expected_values


def print_solution_distribution(predict, data, expected_values, ids, writer):

    data, expected_values = preprocess_data(data, expected_values, remove_high_rr=False)

    for id, (i, row) in zip(ids[1:], enumerate(data[1:])):
        prediction = predict(np.array(row).astype(np.float))

        solution_row = [id]
        solution_row.extend(prediction)
        writer.writerow(solution_row)

        if i % 1000 == 0:
            logger.info("Completed row %d" % i)

def wrap_threshold_distribtuion(prediction):
    return [0 if x < prediction else 1 for x in range(70)]

def wrap_sigmoid_distribution(prediction, length=70):
    xs = np.arange(length)
    return 1. / (1 + np.exp(-(xs - prediction)))

def train_rr1(data, expected_values):
    data, expected_values = preprocess_data(data, expected_values, remove_high_rr=False)
    return lambda x: wrap_sigmoid_distribution(np.float(x[np.where(data[0]=='RR1')[0][0]]))


def train_lg(data, expected_values):
    data, expected_values = preprocess_data(data, expected_values, remove_high_rr=False, delete_features=False)
    logger.info("DATA: %s" % data[:1])
    X = np.asarray(data[1:], 'float')
    Y = np.round(np.asarray(expected_values, 'float'))
    #Y = expected_values
    S = zip(X, Y)
    S = filter(lambda (x,y): (y != 0) and (y < 70), S)
    S = filter(lambda (x,y): abs(x[1] - y) < 3, S)
    X, Y = zip(*S)
    X = np.array(X)
    X = X[:,[1,2,3,4]]
    clf = linear_model.LogisticRegression()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    clf.fit(X_train,Y_train)

    logger.info("Logistic regression using RBM features:\n%s\n" % (
            metrics.classification_report(
            Y_test,
            clf.predict(X_test))))


    return lambda x: np.cumsum(fix_dist(x, clf))


def fix_dist(row, clf):
    x = np.asarray(row[[1,2,3,4]], 'float')
    dist = clf.predict_proba(x).flatten()
    d = dict(zip(clf.classes_, dist))
    y = [d.get(i, 0) for i in range(70)]
    return y


def train_svr(data, expected_values):
    data, expected_values = preprocess_data(data, expected_values, remove_high_rr=False)
    logger.info("Starting feature reduction.")
    data = reduce_features(data[1:], NUMBER_OF_FEATURES)
    logger.info("Done with feature reduction.")

    logger.info("Starting SVR training.")
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(data, expected_values)
    logger.info("Done with SVR training.")
    return lambda x: wrap_threshold_distribtuion(clf.predict(x))

def train_nn(data, expected_values):
    data, expected_values = preprocess_data(data, expected_values, remove_high_rr=False)
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
    parser.add_argument('method', choices=['rr1', 'svr', 'nn', 'tc', 'lg'],
                        help=("Method to be used to generate the solution."))
    parser.add_argument('--output', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help=("path to an output file, "
                              "defaults to stdout"))
    parser.add_argument('--export_params', type=argparse.FileType('w'),
                        default=None,
                        help=("File to write trained parameters to."))
    parser.add_argument('--import_params', type=argparse.FileType('r'),
                        default=None,
                        help=("File to load trained parameters from."))
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help=("Use if you don't know expected values."))
    parser.set_defaults(evaluate=False)

    args = parser.parse_args()
    produce_solution(args)
