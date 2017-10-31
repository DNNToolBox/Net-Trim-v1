"""
  Feel free to redistribute and/or modify this software as long as you
  make a reference to the original source code and cite the following paper

  A. Aghasi, A. Abdi, N. Nguyen, and J. Romberg, "Net-Trim: Convex Pruning of
  Deep Neural Networks with Performance Guarantee," NIPS 2017

  Created by: Afshin Abdi, Georgia Tech
  Email: abdi@ece.gatech.edu
  Created: Fall 2017
"""

import tensorflow as tf
import numpy as np
import scipy


class NetTrimSolver:
    def __init__(self):
        self._graph = None
        self._sess = None
        self._initializer = None

        # inputs of the graph
        self._L = None
        self._U = None
        self._A = None
        self._q = None
        self._c = None

        # optimization parameters
        self._rho = None
        self._alpha = None

        # initial values of x, z, u
        self._init_z = None
        self._init_u = None

        # outputs
        self._x = None
        self._z = None
        self._u = None
        self._dx = None

    def create_graph(self, unroll_number=10):
        self._graph = tf.Graph()

        with self._graph.as_default():
            # inputs to the graph
            self._L = tf.placeholder(tf.float64)
            self._U = tf.placeholder(tf.float64)
            self._A = tf.placeholder(tf.float64)
            self._q = tf.placeholder(tf.float64)
            self._c = tf.placeholder(tf.float64)

            # initial values for x, z and u
            self._init_z = tf.placeholder(tf.float64)
            self._init_u = tf.placeholder(tf.float64)

            # optimization parameters
            self._rho = tf.placeholder(dtype=tf.float64)
            self._alpha = tf.placeholder(dtype=tf.float64)

            # dummy variables
            At = tf.transpose(self._A)

            z = self._init_z
            u = self._init_u

            # first iteration to compute x with initial values of 0 for other variables
            # compute x
            _1 = tf.subtract(self._c, tf.add(u, z))  # c-u-z
            _2 = tf.multiply(self._rho, tf.matmul(At, _1))  # rho*A'*(c-u-z)
            _3 = tf.subtract(_2, self._q)  # rho*A'*(c-u-z)-q
            _4 = tf.matrix_triangular_solve(self._L, _3, lower=True)
            x = tf.matrix_triangular_solve(self._U, _4, lower=False)
            x_prev = x

            for i in range(unroll_number):
                Ax = tf.matmul(self._A, x)
                c_Ax = tf.subtract(self._c, Ax)

                # update z
                _1 = tf.multiply(self._alpha, c_Ax)  # alpha*(c-A*x)
                _2 = tf.multiply(1 - self._alpha, z)  # (1-alpha)*z_prev
                tmp = tf.subtract(tf.add(_1, _2), u)  # alpha*(c-A*x) + (1-alpha)*z_prev - u
                z = tf.maximum(tmp, 0)  # z = max(alpha*(c-A*x) + (1-alpha)*z_prev, 0)

                # update u
                u = tf.subtract(z, tmp)  # z - (alpha*(c-A*x) + (1-alpha)*z_prev - u)

                # update x
                x_prev = x
                _1 = tf.subtract(self._c, tf.add(u, z))  # c-u-z
                _2 = tf.multiply(self._rho, tf.matmul(At, _1))  # rho*A'*(c-u-z)
                _3 = tf.subtract(_2, self._q)  # rho*A'*(c-u-z)-q
                _4 = tf.matrix_triangular_solve(self._L, _3, lower=True)
                x = tf.matrix_triangular_solve(self._U, _4, lower=False)

            self._dx = tf.norm(tf.subtract(x, x_prev))
            self._x = x
            self._z = z
            self._u = u

            self._initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=self._graph)  # ,config=tf.ConfigProto(log_device_placement=True))

    def run(self, X, y, rho, alpha, lmbda, num_iterations=100):
        lmbda = 1 / lmbda
        y = np.reshape(y, newshape=(1, -1))  # make sure y is a row vector
        N = X.shape[0]

        if y.shape[1] != X.shape[1]:
            raise ValueError("Dimensions of input data, X & y, are not consistent.")

        Omega = np.where(y > 1e-6)[1]
        Omega_c = np.where(y <= 1e-6)[1]

        Q = lmbda * np.matmul(X[:, Omega], np.transpose(X[:, Omega]))
        q = -lmbda * np.matmul(X, np.transpose(y))
        P = X[:, Omega_c]
        P = P.transpose()
        c = np.zeros((len(Omega_c), 1))

        Q = np.kron([[1, -1], [-1, 1]], Q)
        q = 1 / 2 + np.append(q, -q, axis=0)
        P = np.append(P, -P, axis=1)

        A = np.append(P, -np.eye(2 * N, 2 * N), axis=0)
        c = np.append(c, np.zeros((2 * N, 1)), axis=0)

        # The ADMM part of the code
        L = np.linalg.cholesky(Q + rho * np.matmul(A.transpose(), A))
        U = L.transpose()

        z = np.zeros((len(c), 1))
        x = np.zeros((2 * N, 1))
        u = np.zeros((len(c), 1))

        self._sess.run(self._initializer)
        cnt = 0
        for cnt in range(num_iterations):
            feed_dict = {self._L: L, self._U: U, self._A: A, self._q: q, self._c: c, self._rho: rho,
                         self._alpha: alpha, self._init_z: z, self._init_u: u}

            dx, x, z, u = self._sess.run([self._dx, self._x, self._z, self._u], feed_dict=feed_dict)
            if np.linalg.norm(dx) < 1e-3:
                break

        w = x[0:N] - x[N:]
        w[np.abs(w) < 1e-3] = 0
        w = w.squeeze()

        return w, cnt


def net_trim_solver_np(X, y, rho, alpha, lmbda, num_iterations=1000):
    lmbda = 1 / lmbda
    y = np.reshape(y, newshape=(1, -1))  # make sure y is a row vector
    N = X.shape[0]

    if y.shape[1] != X.shape[1]:
        raise ValueError("Dimensions of input data, X & y, are not consistent.")

    Omega = np.where(y > 1e-6)[1]
    Omega_c = np.where(y <= 1e-6)[1]

    Q = lmbda * np.matmul(X[:, Omega], np.transpose(X[:, Omega]))
    q = -lmbda * np.matmul(X, np.transpose(y))
    P = X[:, Omega_c]
    P = P.transpose()
    c = np.zeros((len(Omega_c), 1))

    Q = np.kron([[1, -1], [-1, 1]], Q)
    q = 1 / 2 + np.append(q, -q, axis=0)
    P = np.append(P, -P, axis=1)

    A = np.append(P, -np.eye(2 * N, 2 * N), axis=0)
    c = np.append(c, np.zeros((2 * N, 1)), axis=0)

    # The ADMM part of the code
    L = np.linalg.cholesky(Q + rho * np.matmul(A.transpose(), A))
    U = L.transpose()

    z = np.zeros((len(c), 1))
    u = np.zeros((len(c), 1))

    _1 = rho * np.matmul(A.T, c) - q  # rho*A'*(c-u-z)-q
    _2 = scipy.linalg.solve_triangular(L, _1, lower=True)  # np.linalg.solve(L, _3)
    x = scipy.linalg.solve_triangular(U, _2, lower=False)  # np.linalg.solve(U, _4)

    cnt = 0
    for cnt in range(num_iterations):
        c_Ax = c - np.matmul(A, x)

        # update z
        tmp = alpha * c_Ax + (1 - alpha) * z - u  # alpha*(c-A*x) + (1-alpha)*z_prev - u
        z = np.maximum(tmp, 0)  # z = max(alpha*(c-A*x) + (1-alpha)*z_prev, 0)

        # update u
        u = z - tmp  # z - (alpha*(c-A*x) + (1-alpha)*z_prev - u)

        # update x
        x_prev = x
        _1 = rho * np.matmul(A.T, c - u - z) - q  # rho*A'*(c-u-z)-q
        _2 = scipy.linalg.solve_triangular(L, _1, lower=True)
        x = scipy.linalg.solve_triangular(U, _2)

        if np.linalg.norm(x - x_prev) < 1e-3:
            break

    w = x[0:N] - x[N:]
    w[np.abs(w) < 1e-3] = 0

    return w, cnt
