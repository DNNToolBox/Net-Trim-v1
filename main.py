from tensorflow.examples.tutorials.mnist import input_data

import NetTrimSolver as nt_solver
import GeneralSoftmaxModel as nn_model

import time
import numpy as np
import scipy.io as sio
import itertools
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_neural_network(model_size, file_name):
    mnist = input_data.read_data_sets("Z:/Data/MNIST/", one_hot=True)

    batch_size = 64
    epochs = 1
    sample_number = 10000
    sample_number = (sample_number // batch_size - 1) * batch_size

    train_x = mnist.train.images[0:(sample_number + batch_size), :]
    train_y = mnist.train.labels[0:(sample_number + batch_size), :]
    model = nn_model.GeneralSoftmaxModel(model_size=model_size, regulize=True, regularization_gain=5e-4)

    model.initialize()

    acc = model.compute_accuracy(mnist.test.images, mnist.test.labels)
    print('initial accuracy = {0:5.3f}'.format(acc))
    for epoch in range(epochs):
        for train_index in range(0, sample_number, batch_size):
            batch_x = train_x[range(train_index, train_index + batch_size), :]
            batch_y = train_y[range(train_index, train_index + batch_size), :]

            model.train(batch_x, batch_y, 0.5)

        acc = model.compute_accuracy(mnist.validation.images, mnist.validation.labels)
        print('iteration {0:2d}: accuracy = {1:5.3f}'.format(epoch, acc))

    acc = model.compute_accuracy(mnist.test.images, mnist.test.labels)
    print('final accuracy = {0:5.3f}'.format(acc))

    # get signals of the model
    nn_weights, nn_biases = model.get_weights()
    model_signals = model.compute_signals(train_x)

    # ====================================================================
    # store signals and parameters in a .mat file
    # signals
    mat_signals = {}
    for i, X in zip(itertools.count(), model_signals):
        mat_signals['X{}'.format(i)] = X

    # coefficients W
    mat_weights = {}
    for i, W in zip(itertools.count(), nn_weights):
        mat_weights['W{}'.format(i)] = W

    # biases b
    mat_biases = {}
    for i, b in zip(itertools.count(), nn_biases):
        mat_biases['b{}'.format(i)] = b

    num_layers = len(model_size)
    data = {'N': num_layers, **mat_signals, **mat_weights, **mat_biases}
    sio.savemat(file_name, data)
    # ====================================================================

    return data


if __name__ == '__main__':
    data_file_name = 'data.mat'
    if os.path.isfile(data_file_name):
        data = sio.loadmat(data_file_name)
    else:
        data = train_neural_network(model_size=[784, 300, 10], file_name=data_file_name)

    solver_tf = nt_solver.NetTrimSolver()
    solver_tf.create_graph(unroll_number=100)

    X = data['X0']
    Y = data['X1']
    X = X.transpose()
    Y = Y.transpose()

    # append 1 to the last row of X for model y = ReLU(W'x+b)
    X = np.append(X, np.ones(shape=(1, X.shape[1])), axis=0)

    original_W = data['W0']
    original_b = data['b0']
    refined_W = original_W.copy()
    refined_b = original_b.copy()
    total_time = 0
    for i in range(Y.shape[0]):
        # start = time.time()
        # w_np, num_iter_np = nt_solver.net_trim_solver_np(X=X, y=Y[i, :], rho=5, alpha=1.8, lmbda=4, num_iterations=500)
        # time_np = time.time() - start
        # print('numpy-based execution time:', time_np)

        start = time.time()
        w_tf, num_iter_tf = solver_tf.run(X=X, y=Y[i, :], rho=5, alpha=1.8, lmbda=4, num_iterations=5)
        elapsed_time = time.time() - start
        print('tensorflow-based execution time:', elapsed_time)
        refined_W[:, i] = w_tf[:-1]
        refined_b[0, i] = w_tf[-1]

        total_time += elapsed_time

        # print(num_iter_np, np.linalg.norm(w_np))
        # print(num_iter_tf, np.linalg.norm(w_tf - w_np))
        # print('original weight = ', data['W0'][0:10, 0])
        # print('refined weight = ', w_np[0:10])
        # print('refined weight = ', w_tf[0:10])

    print('number of non-zero values in the original weight matrix = ', np.count_nonzero(original_W == 0))
    print('number of non-zero values in the refined weight matrix = ', np.count_nonzero(refined_W == 0))
    print('total elapsed time = ', total_time)