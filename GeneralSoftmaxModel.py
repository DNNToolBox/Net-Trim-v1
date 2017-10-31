import tensorflow as tf


class GeneralSoftmaxModel:
    """ General fully connected softmax neural network for classification of input data.
        It contains multiple hidden layers (provided by the user) with 
        ReLU activation functions and the output layer is Softmax. Dropout is added for training as well.  """

    def __init__(self, initial_weights=None, model_size=None, training_algorithm='GD', learning_rate=0.1,
                 training_parameter=0.9, regulize=False, regularization_gain=5e-4, computation_precision=tf.float32):

        self._computation_precision = computation_precision
        self._graph = None
        self._sess = None
        self._trainOp = None
        self._initializer = None
        self._accuracy = None

        # parameters of the neural network
        self._input = None
        self._output = None
        self._target = None
        self._gradients = None
        self._keep_prob = None
        self._nn_signals = []
        self._nn_weights = []
        self._nn_biases = []

        self._num_parameters = 0

        if initial_weights is not None:
            self._create_initialized_graph(initial_weights)
        elif model_size is not None:
            self._create_random_graph(model_size)
        else:
            raise ValueError('Network size is not given.')

        self._define_optimizer(training_algorithm, learning_rate, training_parameter, regulize, regularization_gain)
        self._num_parameters = len(self._nn_weights)
        self._sess = tf.Session(graph=self._graph)

    # =========================================================================
    # Create the computational graph, including:
    #      1- Neural Network (placeholders, weights and output)
    #      2- Initializer
    #      3- Values and Gradients of the parameters of the NN
    #      4- Optimizer
    def _create_initialized_graph(self, initial_weights):
        if self._graph is not None:
            return

        num_layers = len(initial_weights)
        input_len = initial_weights[0][0].shape[0]
        output_len = initial_weights[-1][0].shape[1]

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._x = tf.placeholder(self._computation_precision, [None, input_len])
            self._target = tf.placeholder(self._computation_precision, [None, output_len])
            self._keep_prob = tf.placeholder(tf.float32, 1)

            # create layers
            self._nn_signals = [self._x]
            self._nn_weights = []
            self._nn_biases = []

            out_dim = input_len  # number of output nodes from previous layer
            output_signal = self._x
            output_signal_drop = self._x  # no dropout for the input signal
            for h in range(num_layers):
                init_w = initial_weights[h][0]
                init_b = initial_weights[h][1]
                if (init_w.shape[0] != out_dim) or (init_w.shape[1] != init_b.shape):
                    raise ValueError('Inconsistent dimensions for initial weights.')

                W = tf.Variable(init_w)
                b = tf.Variable(init_b)

                output_signal = tf.matmul(output_signal_drop, W) + b
                if h == num_layers - 1:
                    # final layer is Softmax
                    output_signal = tf.nn.softmax(output_signal)
                else:
                    # hidden layers are ReLU
                    output_signal = tf.nn.relu(output_signal)

                output_signal_drop = tf.nn.dropout(output_signal, self._keep_prob)
                out_dim = init_w.shape[1]

                self._nn_weights += [W]
                self._nn_biases += [b]
                self._nn_signals += [output_signal]

            self._output = output_signal

    # =========================================================================
    def _create_random_graph(self, model_size):
        if self._graph is not None:
            return

        num_layers = len(model_size) - 1
        input_len = model_size[0]
        output_len = model_size[-1]

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._x = tf.placeholder(self._computation_precision, [None, input_len])
            self._target = tf.placeholder(self._computation_precision, [None, output_len])
            self._keep_prob = tf.placeholder(self._computation_precision)

            # create layers
            self._nn_signals = [self._x]
            self._nn_weights = []
            self._nn_biases = []

            output_signal = self._x
            output_signal_drop = self._x  # no dropout for the input signal
            for h in range(num_layers):
                in_dim = model_size[h]  # number of input nodes to the hidden layer
                out_dim = model_size[h + 1]  # number of output nodes of the hidden layer

                W = tf.Variable(tf.truncated_normal([in_dim, out_dim], 0, 0.1, dtype=self._computation_precision))
                b = tf.Variable(0.1 * tf.ones([out_dim], dtype=self._computation_precision))

                output_signal = tf.matmul(output_signal_drop, W) + b
                if h == num_layers - 1:
                    # final layer is Softmax
                    output_signal = tf.nn.softmax(output_signal)
                else:
                    # hidden layers are ReLU
                    output_signal = tf.nn.relu(output_signal)

                output_signal_drop = tf.nn.dropout(output_signal, self._keep_prob)

                self._nn_weights += [W]
                self._nn_biases += [b]
                self._nn_signals += [output_signal]

            self._output = output_signal

    # =========================================================================
    def _define_optimizer(self, training_algorithm, learning_rate, training_parameter=0.9, regulize=False, gain=1e-4):
        with self._graph.as_default():
            # the cost function
            cross_entropy = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(logits=self._output, onehot_labels=self._target))

            if regulize:
                # regulization of the W coefficients. it does not include the biases, b
                regulizer_cost = 0
                for w in self._nn_weights:
                    regulizer_cost += tf.nn.l2_loss(w)

                cost = cross_entropy + gain * regulizer_cost
            else:
                cost = cross_entropy

            # =================================================================
            # define the appropriate optimizer to use
            if (training_algorithm == 0) or (training_algorithm == 'GD'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif (training_algorithm == 1) or (training_algorithm == 'RMSProp'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif (training_algorithm == 2) or (training_algorithm == 'Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif (training_algorithm == 3) or (training_algorithm == 'AdaGrad'):
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            elif (training_algorithm == 4) or (training_algorithm == 'AdaDelta'):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            elif (training_algorithm == 5) or (training_algorithm == 'Momentum'):
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=training_parameter)
            else:
                raise ValueError("Unknown training algorithm.")

            # =================================================================
            # training and initialization operators
            parameter_list = self._nn_weights + self._nn_biases
            gv = optimizer.compute_gradients(cost, var_list=parameter_list)
            self._gradients = [g for (g, _) in gv]

            self._trainOp = optimizer.minimize(cost, var_list=parameter_list)
            self._initializer = tf.global_variables_initializer()

            # =================================================================
            # update (assign) operator for the parameters of the NN model
            self._weight_assign_op = []
            self._weight_placeholders = ()
            self._bias_assign_op = []
            self._bias_placeholders = ()

            for w in self._nn_weights:
                holder = tf.placeholder(dtype=self._computation_precision, shape=w.get_shape())
                assign_op = w.assign(holder)
                self._weight_assign_op.append(assign_op)
                self._weight_placeholders = self._weight_placeholders + (holder,)

            for b in self._nn_biases:
                holder = tf.placeholder(dtype=self._computation_precision, shape=b.get_shape())
                assign_op = b.assign(holder)
                self._bias_assign_op.append(assign_op)
                self._bias_placeholders = self._bias_placeholders + (holder,)

    # =========================================================================
    # Number of parameter pairs (W, b)
    @property
    def num_parameters(self):
        return self._num_parameters

    # =========================================================================
    # Accuracy rate of the NN (defined as property)
    @property
    def accuracy(self):
        if self._accuracy is None:
            with self._graph.as_default():
                matches = tf.equal(
                    tf.argmax(self._target, 1), tf.argmax(self._output, 1))
                self._accuracy = tf.reduce_mean(tf.cast(matches, self._computation_precision))

        return self._accuracy

    # =========================================================================
    # Initialize the computation graph
    def initialize(self):
        if self._initializer is not None:
            self._sess.run(self._initializer)
        else:
            raise ValueError('Initializer has not been set.')

    # =========================================================================
    # One iteration of the training algorithm with input data
    def train(self, batch_x, batch_y, keep_prob=1.0):
        if self._trainOp is not None:
            self._sess.run(self._trainOp,
                           feed_dict={self._x: batch_x, self._target: batch_y, self._keep_prob: keep_prob})
        else:
            raise ValueError('Training algorithm has not been set.')

    # =========================================================================
    # Get values of the parameters of the NN
    def get_weights(self):
        return self._sess.run(self._nn_weights), self._sess.run(self._nn_biases)

    # =========================================================================
    # Set values of the parameters of the NN
    def set_weights(self, new_weights=None, new_biases=None):
        if new_weights is not None:
            self._sess.run(self._weight_assign_op, {self._weight_placeholders: tuple(new_weights)})
        if new_biases is not None:
            self._sess.run(self._bias_assign_op, {self._bias_placeholders: tuple(new_biases)})

    # =========================================================================
    # Compute the gradients of the parameters of the NN for the given input
    def compute_gradients(self, x, target):
        return self._sess.run(self._gradients,
                              feed_dict={self._x: x, self._target: target})

    # =========================================================================
    # Compute the accuracy of the NN using the given inputs
    def compute_accuracy(self, x, target):
        return self._sess.run(self.accuracy,
                              feed_dict={self._x: x, self._target: target, self._keep_prob: 1.0})

    # =========================================================================
    # Compute the signals in the NN
    def compute_signals(self, x):
        return self._sess.run(self._nn_signals, feed_dict={self._x: x, self._keep_prob: 1.0})
