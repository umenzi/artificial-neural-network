import numpy as np
import random


# Loss functions
class QuadraticLoss(object):

    @staticmethod
    def fn(a, y):
        """
        Return the loss associated with an output ``a`` and desired output ``y``. I.e., how well a matches y.
        """

        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.
        """

        return (a - y) * SigmoidActivation.delta(z)


class CrossEntropyLoss(object):
    """
    The implementation of the cross-entropy loss function is based on
    the formula: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    """

    @staticmethod
    def fn(a, y):
        """
        Return the loss associated with an activation output ``a`` and desired output
        ``y``. I.e., how well a matches y.
        Note that np.nan_to_num is used to ensure numerical stability. In particular,
        if both ``a`` and ``y`` have a 1.0 in the same slot, then the expression
        (1-y) * np.log(1-a) returns nan. The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        epsilon = 1e-12
        a = np.clip(a, epsilon, 1 - epsilon)  # ensure ``a`` is within (0, 1)

        return -np.sum(y * np.log(a))

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.
        The parameter ``z`` is included to keep consistency with other loss functions, but it is not used.
        """
        return a - y


# Activation functions
class SigmoidActivation(object):

    @staticmethod
    def fn(z):
        """
        The sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def delta(z):
        """
        The derivative of the sigmoid function.
        """
        return SigmoidActivation.fn(z) * (1 - SigmoidActivation.fn(z))


class SoftMaxActivation(object):

    @staticmethod
    def fn(x):
        """
        Compute softmax values for each set of scores in x.
        """
        # e_x = np.exp(x - np.max(x))
        # return e_x / e_x.sum(axis=0)
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

    @staticmethod
    def delta(x):
        """
        Computes the derivative of the softmax activation function.
        :param x: Input data,
        :return: the derivative
        """

        x = SoftMaxActivation.fn(x).reshape(-1, 1)
        return np.diagflat(x) - np.dot(x, x.T)


class ReLUActivation(object):

    @staticmethod
    def fn(z):
        """
        The ReLU function.
        """
        return np.maximum(z, 0.0)

    @staticmethod
    def delta(z):
        """
        The derivative of the ReLU function.
        """
        return 1.0 * (z > 0.0)


class LeakyReLUActivation(object):
    """
    By default, we use a gradient of 0.01, as used by Maas et al.
    when this activation function was introduced.
    """

    @staticmethod
    def fn(z, gradient=0.01):
        """
        The leaky ReLU function.
        """
        return np.where(z > 0, z, z * gradient)

    @staticmethod
    def delta(z, gradient=0.01):
        """
        The derivative of the leaky ReLU function.
        """
        return np.where(z > 0.0, 1.0, gradient)


# The single Perceptron class used in section 1.2

class Perceptron:
    def __init__(self, input_size, bias=-1, learning_rate=0.1, epochs=10):
        # The weight vector. We increase the input size by one to include the bias.
        # We randomize the weights in the range (-0.5, 0.5)
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=input_size)
        # self.W = np.zeros(input_size)
        # Bias
        self.bias = bias
        # Number of epochs
        self.epochs = epochs
        # Learning rate
        self.learning_rate = learning_rate

    # The activation function. We use the step function as in the lectures
    def activation_step_fn(self, x):
        return 1 if x >= 0 else 0

    def compute_error(self, y, y_hat):
        return y_hat - y

    # Run an input through the perceptron and return an output
    def predict(self, user_input):
        # Compute the inner product.
        weighted_sum = self.weights.T.dot(user_input) + self.bias
        # Apply the activation function.
        output = self.activation_step_fn(weighted_sum)
        return output

    # Perceptron learning algorithm
    def train(self, inputs, labels):
        # List that tracks the errors per epoch
        error_list = []
        # We keep updating the weights for a number of epochs, and iterate through the entire training set
        for _ in range(self.epochs):
            cur_error = 0
            for user_input, label in zip(inputs, labels):
                # Calculate prediction
                prediction = self.predict(user_input)
                # Calculate error
                error = self.compute_error(prediction, label)
                cur_error += abs(error)
                # Update the weight of the perceptron
                self.weights += self.learning_rate * user_input * error
                # Update the bias of the perceptron
                self.bias += self.learning_rate * error
            error_list.append(cur_error)
        return error_list


# The MLP Network used by sections 1.3+

class Layer:
    def __init__(self, size_input, size_output, loss_function=CrossEntropyLoss, is_output_layer=False,
                 is_input_layer=False, activation_function=ReLUActivation):
        """
        The layer of an MLP network.

        :param size_input: How many units does the layer have.
        :param loss_function: The loss function used by the layer.
        :param is_output_layer: If the layer is an output layer (we calculate the backprop differently
                if so).
        :param is_input_layer: If the layer is input layer (we initialize the bias differently
                if so).
        :param activation_function: The activation function used by the layer.
        """

        # Check the layer is not input and output and the same time
        assert not all([is_input_layer, is_output_layer])

        self.size = (size_input, size_output)

        self.bias = None
        # If input layer, we don't have any bias (only in further layers)
        if is_input_layer:
            self.bias = np.zeros(shape=(self.size[1], 1))

        self.is_output_layer = is_output_layer

        self.weight = None

        self.initializer()

        self.loss_function = loss_function

        self.activation_function = activation_function

        # For backpropagation
        self.weighted_sum = None
        self.activations = []

    def initializer(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 divided by the square root of the number of
        weights connecting to the same neuron.

        Initialize the biases using a Gaussian distribution with mean zero and
        standard deviation 1, if NOT the input layer. Else, we won't set any biases
        (i.e., kept to 0), since biases should only be used to compute the outputs
        from later layers.
        """

        # We only randomize if the layer is not the input layer
        if self.bias is None:
            self.bias = np.random.randn(self.size[1], 1)

        # Normal initialization: self.weight = np.random.randn(self.size[1], self.size[0])

        if self.is_output_layer:
            # We use Softmax, Sigmoid or Tanh in the output layer. So, we choose "Xavier initialization"
            self.weight = np.random.randn(self.size[1], self.size[0]) / np.sqrt(self.size[1])
        else:
            # In hidden layers, we prefer to use ReLU, LReLU, ELU, etc. So, we choose "He initialization"
            self.weight = np.random.randn(self.size[1], self.size[0]) * np.sqrt(2.0 / self.size[1])

    def get_weights(self):
        return np.mean(np.sum(self.weight))

    def feedforward(self, a, is_back_propagation=False):
        """
        Given an input a for the network, returns the corresponding output.
        In mathematical terms, we compute ``a′ = σ(w * a + b)``.

        :param a: the input.
        :param is_back_propagation: if we are performing the feedforward step
                as part of the backpropagation algorithm (this method is also called
                for predicting unknown data).
        :return: the output of the network.
        """

        z = np.dot(self.weight, a) + self.bias
        activation = self.activation_function.fn(z)

        # If backpropagation, we store the last weighted sum and activations that have gone through this layer
        if is_back_propagation:
            self.weighted_sum = z
            self.activations = [a, activation]

        return activation

    def backpropagation(self, y, delta, previous_weight=None):
        """
        Performs a single backprop step, for one single training sample.

        :param previous_weight: The weight of the following layer
        (who performs backpropagation on this layer).
        :param y: The training sample.
        :param delta: The derivative of the loss function.
        :return: The derivative of the loss function, nabla_b, nabla_w.
        """

        if self.is_output_layer:
            delta = self.loss_function.delta(self.weighted_sum, self.activations[-1], y)

            # if not isinstance(type(self.loss_function), CrossEntropyLoss):
            #     delta *= self.activation_function.delta(self.weighted_sum)
            nabla_b = delta
            nabla_w = np.dot(delta, self.activations[-2].transpose())

        else:
            sp = self.activation_function.delta(self.weighted_sum)
            delta = np.dot(previous_weight.transpose(), delta) * sp
            nabla_b = delta
            nabla_w = np.dot(delta, self.activations[-2].transpose())

        return delta, nabla_b, nabla_w, self.weight


class ANN:
    def __init__(self, sizes, loss_function=CrossEntropyLoss,
                 hidden_activation_function=ReLUActivation, output_activation_function=SoftMaxActivation):
        """
        An MLP network that supports many layers each with different sizes, choosing a specific loss function
        and activation function.

        Note that the first layer is assumed to be an input layer, and by convention we won't set any biases for
        those neurons, since biases are only ever used in computing the outputs from later layers.

        :param sizes: the size of each layer. For example, if the list
                was ``[10, 9, 7]`` then it would be a three-layer network, with the
                first (input) layer containing ``10` neurons, the second (hidden) layer ``9`` neurons,
                and the third (output) layer ``7`` neuron.
        :param loss_function: the loss function used by the network.
        :param hidden_activation_function: the activation function used by the network at the hidden layers.
        :param output_activation_function: the activation function used by the network at the output layer.
        """

        # Check sizes list has at least 3 layers (input, output, hidden),
        # and its values are non-negative
        assert all(size > 0 for size in sizes)
        assert len(sizes) > 2

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.loss_function = loss_function

        self.layers = []

        # We do not include the input layer here, as it is simply a dummy layer that simply
        # contains the given input.
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.layers.append(Layer(x, y, loss_function=loss_function, activation_function=hidden_activation_function))

        # The last layer is the output layer
        self.layers[-1].is_output_layer = True
        self.layers[-1].activation_function = output_activation_function

    def feedforward(self, a):
        """
        Given an input a for the network, returns the corresponding output.
        In mathematical terms, we compute ``a′ = σ(w * a + b)``.

        :param a: the input.
        :return: the output of the network.
        """

        for layer in self.layers:
            a = layer.feedforward(a)
        return a

    def predict(self, a):
        """
        Predicts the output of a given input.

        :param a: The input to the input layer.
        :return: The predicted output.
        """

        # We need to increase the output by 1 because the targets start from
        # 1, but our network starts from 0 (as Python does)
        y_hat = np.argmax(self.feedforward(a)) + 1
        return y_hat

    def sgd(self, training_data, epochs=100, batch_size=10, learning_rate=0.1,
            reg_param=0.0, n_iter_no_change=15, tol=1e-4, adam_active=False, step_decay=False,
            early_stopping=False, validation_data=None, print_progress=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.

        In each epoch, it starts by randomly shuffling the training data, and then partitions it
        into mini-batches of the appropriate size. This is an easy way of sampling randomly
        from the training data. Then for each mini_batch we apply a single step of gradient descent.

        :param training_data: list of tuples ``(x, y)`` representing training inputs and corresponding desired outputs.
        :param epochs: the number of epochs to train for.
        :param batch_size: the size of the mini-batches to use when sampling. If 1 (default),
                stochastic gradient descent is performed. If = len(training_data), the whole batch is used directly.
        :param learning_rate: the learning rate.
        :param validation_data: Optional argument. If provided, then the network will be evaluated against
                the test data after each epoch.
        :param print_progress: Optional argument. Partial progress per epoch is printed out on the console.
                As a result, the performance may be slowed down.
        :param reg_param: the regularization rate (for regularization, optional parameter)
        :param n_iter_no_change: patience of the early stopping algorithm
        :param tol: min difference between losses for which early stopping becomes impatient
        :param adam_active: If we compute the Adam algorithm instead of default SGD
        :param early_stopping: If we compute early stopping.
        :param step_decay: If we compute step decay (update the learning rate every few epochs)
        :return: If test data is provided, returns the errors per epoch.
        """

        if adam_active or step_decay:
            assert not (adam_active and step_decay)

        decay_rate = learning_rate / epochs

        prev_error = float('-inf')

        # Params for early stopping. Implementation based on
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html
        # and Deep Learning's book
        best_val_loss = float('inf')
        early_stopping_layers = self.layers
        early_stopping_epochs = 0
        early_stopping_stop = 0

        train_errors = []
        val_errors = []
        train_loss = []
        val_loss = []

        for epoch in range(epochs):
            random.shuffle(training_data)

            cache1 = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            cache2 = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

            t = 1

            mini_batches = [
                training_data[i:i + batch_size]
                for i in range(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, learning_rate, reg_param=reg_param, cache1=cache1,
                                  t=t, cache2=cache2, n=len(training_data),
                                  adam_active=adam_active)
                t += 1

            cur_train_error = self.evaluate(training_data)
            cur_train_loss = self.compute_loss(training_data, 0)

            if validation_data:
                cur_val_error = self.evaluate(validation_data)
                cur_val_loss = self.compute_loss(validation_data, 0)
                train_errors.append(cur_train_error)
                val_errors.append(cur_val_error)
                train_loss.append(cur_train_loss)
                val_loss.append(cur_val_loss)
                if print_progress:
                    print("Epoch {0} -> Train accuracy: {1} / {2}, Validation accuracy: {3} / {4}".format(
                        epoch, cur_train_error, len(training_data), cur_val_error, len(validation_data)
                    ))
                    print("Epoch {0} -> Train loss: {1}, Validation loss: {2}".format(
                        epoch, cur_train_loss, cur_val_loss
                    ))

                # Early stopping
                if early_stopping:
                    cur_val_loss = self.compute_loss(validation_data, reg_param=reg_param)
                    if best_val_loss - cur_val_loss >= tol:
                        early_stopping_layers = self.layers
                        early_stopping_epochs = epoch
                        best_val_loss = cur_val_loss
                        early_stopping_stop = 0
                    else:
                        early_stopping_stop += 1
                    # if the scores of the model in the last n_iter_no_change stages do not improve by at least tol
                    if early_stopping_stop >= n_iter_no_change:
                        self.layers = early_stopping_layers
                        if print_progress:
                            print("Training stopped by Early stopping: Most optimal model found after",
                                  early_stopping_epochs, " epochs.")
                        return train_errors, val_errors, train_loss, val_loss

            elif print_progress:
                train_errors.append(cur_train_error)
                if print_progress:
                    print("Epoch {0} -> Train accuracy: {1} / {2}".format(
                        epoch, cur_train_error, len(training_data)
                    ))

            if step_decay and prev_error < self.evaluate(validation_data):
                prev_error = self.evaluate(validation_data)
                learning_rate = max(0.001, learning_rate * (1. / (1. + decay_rate * epoch)))

        if early_stopping:
            self.layers = early_stopping_layers
            if print_progress:
                print("Early stopping: Most optimal model found after", early_stopping_epochs, " epochs.")

        return train_errors, val_errors, train_loss, val_loss

    def update_batch(self, batch, learning_rate, reg_param=0.0, cache1=None,
                     cache2=None, t=0, n=1, adam_active=False):
        """
        Update the network's weights and biases according to a single iteration of gradient descent,
        by using backpropagation to a single mini batch.

        It first computes the gradients for every training example in the mini_batch, and then updates the
        weights and biases of all the layers appropriately.

        :param batch: list of tuples ``(x, y)``.
        :param learning_rate: the learning rate.
        :param reg_param: the regularization rate (for regularization, optional parameter)
        :param cache1: Cache for Adam algorithm
        :param cache2: Cache for Adam algorithm
        :param t: Current epoch (used by Adam algorithm)
        :param n: the length of the training data (used for regularization, optional parameter)
        :param adam_active: If we compute the Adam algorithm instead of default SGD
        """

        gradient_b = [np.zeros(b.bias.shape) for b in self.layers]
        gradient_w = [np.zeros(w.weight.shape) for w in self.layers]

        # Perform forward pass and backward pass, accumulating the gradients for each model parameter (weight)
        for x, y in batch:
            delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
            gradient_b = [n_bias + dn_bias for n_bias, dn_bias in zip(gradient_b, delta_gradient_b)]
            gradient_w = [n_weight + dn_weight for n_weight, dn_weight in zip(gradient_w, delta_gradient_w)]

        # Update the parameters with the average gradient (using the learning rate)
        for i, layer in enumerate(self.layers):
            # Update the bias
            layer.bias -= (learning_rate / len(batch)) * gradient_b[i]

            if adam_active:
                # We compute the Adam algorithm, industry standard for updating the
                # learning rate, https://cs231n.github.io/neural-networks-3/#ada
                eps = 1e-8
                beta1 = 0.9
                beta2 = 0.999

                # t is your iteration counter going from one to infinity
                g = gradient_w[i] / len(batch)
                cache1[i] = beta1 * cache1[i] + (1 - beta1) * g
                mt = cache1[i] / (1 - beta1 ** t)
                cache2[i] = beta2 * cache2[i] + (1 - beta2) * (g ** 2)
                vt = cache2[i] / (1 - beta2 ** t)
                res = - learning_rate * mt / (np.sqrt(vt) + eps)
            else:
                # regular gradient descent
                res = - gradient_w[i] * (learning_rate / len(batch))

            # Update the weight
            if reg_param > 0.0:
                # Perform L2 regularization, also known as weight decay or Ridge Regularization
                layer.weight = (1 - learning_rate * (reg_param / n)) * layer.weight + res
            else:
                layer.weight += res

    def backpropagation(self, x, y):
        """
        Computes a backprop step in the whole network, for one training example and only one epoch.

        :param x: The input.
        :param y: The (expected) output.
        :return: A tuple ``(nabla_b, nabla_w)`` representing the gradient for the loss function ``C_x``.
        """

        delta = None
        nabla_b = [np.zeros(b.bias.shape) for b in self.layers]
        nabla_w = [np.zeros(w.weight.shape) for w in self.layers]

        activation = x

        previous_weight = None

        # Feedforward Pass
        for layer in range(self.num_layers - 1):
            activation = self.layers[layer].feedforward(activation, is_back_propagation=True)

        # Backward Pass
        for layer in range(1, self.num_layers):
            delta, nabla_b[- layer], nabla_w[- layer], previous_weight = \
                self.layers[- layer].backpropagation(y, delta, previous_weight)

        return nabla_b, nabla_w

    def evaluate(self, data):
        """
        Evaluates how the network performs for the given data. In other words, if the predicted outcome is
        similar to the expected outcome.

        The neural network's output is assumed to be the index of whichever neuron in the final
        layer has the highest activation (argmax).

        :param data: List of tuples ``(x, y)`` where ``x`` is the input and ``y`` the outcome.
        :return: How many of the outcomes are equivalent to the data.y
        """

        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                   for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def compute_loss(self, data, reg_param):
        """
        Return the total loss for the given data set.

        :param data: the data set.
        :param reg_param: the regularization parameter.
        :return: The total cost
        """

        loss_cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            loss_cost += self.loss_function.fn(a, y) / len(data)

        added = 0.0
        for layer in self.layers:
            added += np.linalg.norm(layer.weight) ** 2
        loss_cost += ((reg_param / len(data)) * added) / 2

        return loss_cost

    def cross_validate_neurons(self, data, k, num_hidden_list, learning_rate=0.1, epochs=100, batch_size=10, eta=0.0):
        """
        Perform k-fold cross-validation to get the mean error of different number of hidden neurons.

        :param eta: the regularization parameter,
        :param data: List of tuples ``(x, y)`` where ``x`` is the input and ``y`` the outcome,
        :param k: the number of folds,
        :param num_hidden_list: a list of the numbers of hidden neurons,
        :param learning_rate: the learning rate,
        :param epochs: the number of epochs,
        :param batch_size: the batch size,
        :return: a list of the mean error of different number of hidden neurons.
        """

        fold_size = len(data) // k
        validation_errors = []

        for num_hidden in num_hidden_list:
            errors = []
            for i in range(k):
                # split the data into training and validation sets
                val_data = data[i * fold_size:(i + 1) * fold_size]
                training_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

                # train the neural network and evaluate it on the validation set
                net = ANN([10, num_hidden, 7])
                cur_train_error, cur_val_error, cur_train_loss, cur_val_loss \
                    = net.sgd(training_data, epochs=epochs, batch_size=batch_size,
                              reg_param=eta, learning_rate=learning_rate, print_progress=False,
                              early_stopping=False, validation_data=val_data, adam_active=False)
                errors.append(cur_val_error)

            # compute the mean validation error for this number of hidden neurons
            mean_error = np.mean(errors) / fold_size * 100
            if num_hidden == 1:
                print("Average Accuracy for learning rate", learning_rate, "is", mean_error)
            else:
                print("Average Accuracy for", num_hidden, "hidden neurons is", mean_error)

            validation_errors.append(mean_error)

        return validation_errors

    def cross_validate_learning_rates(self, data, k, num_hidden, learning_rates, epochs=100, batch_size=10, eta=0.0):
        """
        Perform k-fold cross-validation to get the mean error of different number of hidden neurons.

        :param eta: The regularization parameter
        :param data: List of tuples ``(x, y)`` where ``x`` is the input and ``y`` the outcome.
        :param k: The number of folds
        :param num_hidden: the number of neurons in the hidden layer
        :param learning_rates: the list learning rates
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param eta: eta
        :return: a list of the mean error of different learning rates
        """

        fold_size = len(data) // k
        validation_errors = []

        for learning_rate in learning_rates:
            errors = []
            for i in range(k):
                # split the data into training and validation sets
                val_data = data[i * fold_size:(i + 1) * fold_size]
                training_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

                # train the neural network and evaluate it on the validation set
                net = ANN([10, num_hidden, 7])
                cur_train_error, cur_val_error, cur_train_loss, cur_val_loss \
                    = net.sgd(training_data, epochs=epochs, batch_size=batch_size, early_stopping=False,
                              reg_param=eta, learning_rate=learning_rate, adam_active=True,
                              print_progress=False, validation_data=val_data)
                errors.append(cur_val_error)

            # compute the mean validation error for this number of hidden neurons
            mean_error = np.mean(errors) / fold_size * 100

            print("Average Accuracy for learning rate", learning_rate, "is", mean_error)

            validation_errors.append(mean_error)

        return validation_errors

    def cross_validate_eta(self, data, k, num_hidden, learning_rate, eta_list, epochs=100, batch_size=10):
        """
        Perform k-fold cross-validation to get the mean error of different number of hidden neurons.

        :param data: List of tuples ``(x, y)`` where ``x`` is the input and ``y`` the outcome.
        :param k: The number of folds
        :param num_hidden: the number of neurons in the hidden layer
        :param learning_rate: the learning rate
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param eta_list: list of etas to cross validates
        :return: a list of the mean error of different etas
        """

        fold_size = len(data) // k
        validation_errors = []

        for eta in eta_list:
            errors = []
            for i in range(k):
                # split the data into training and validation sets
                val_data = data[i * fold_size:(i + 1) * fold_size]
                training_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

                # train the neural network and evaluate it on the validation set
                net = ANN([10, num_hidden, 7])
                cur_train_error, cur_val_error, cur_train_loss, cur_val_loss \
                    = net.sgd(training_data, epochs=epochs, batch_size=batch_size, early_stopping=False,
                              reg_param=eta, learning_rate=learning_rate, adam_active=True,
                              print_progress=False, validation_data=val_data)
                errors.append(cur_val_error)

            # compute the mean validation error for this number of hidden neurons
            mean_error = np.mean(errors) / fold_size * 100

            print("Average Accuracy for eta", eta, "is", mean_error)

            validation_errors.append(mean_error)

        return validation_errors
