import numpy as np
from math import asinh


class JordanNeuralNetwork:
    context_layer = 1
    output_layer_neurons = 1
    default_context_value = 0

    def __init__(self,
                 alpha,
                 steps_border,
                 *shapes):
        shapes = [*shapes[:2:], self.output_layer_neurons]
        self.alpha = alpha
        self.steps_porder = steps_border
        self.excellent_example = None
        self.weights = [np.random.rand(rows, columns) for rows, columns in zip(shapes, shapes[1:])]
        self.values = []
        self.biases = [np.array([np.random.ranf() for _ in range(shape)]) for shape in shapes[1:]]
        self.context_weights = np.array([np.random.ranf() for _ in range(shapes[self.context_layer])])
        self.context_value = self.default_context_value
        self.context_bias = np.random.ranf()

    def get_context_synaptic_array(self):
        function_result = [self.context_value * weight for weight in self.context_weights]
        return np.array(function_result)

    def get_output_neuron(self):
        return self.values[-1].sum()

    def set_context_neuron(self):
        current_output_neuron = self.get_output_neuron()
        self.context_value = self.context_activation_function(current_output_neuron)

    def process_component(self, input_layer, weights):
        layer_result = self.values[-1].dot(weights)
        if input_layer == self.context_layer - 1:
            layer_result += self.get_context_synaptic_array()
        layer_result -= self.biases[input_layer]
        self.values.append(layer_result)

    def process(self, input_data):
        self.values.clear()
        self.values.append(np.reshape(input_data, (1, len(input_data))))
        for input_layer, weights in enumerate(self.weights):
            self.process_component(input_layer, weights)
        self.set_context_neuron()
        return self.get_output_neuron()

    def get_example_error_difference(self):
        return self.get_output_neuron() - self.excellent_example

    def get_network_error(self):
        return (self.get_example_error_difference() ** 2) / 2

    def get_hidden_layer_error(self):
        network_error = self.get_example_error_difference()
        hidden_layer_error = [network_error * weight[0] for weight in self.weights[-1]]
        return hidden_layer_error

    def hidden_output_weights_learning(self):
        error_difference = self.get_example_error_difference()
        for index in range(self.weights[-1].shape[0]):
            current_hidden_value = self.values[1][0][index]
            self.weights[-1][index][0] -= self.alpha * error_difference * current_hidden_value

    def output_biases_learning(self):
        network_error = self.get_example_error_difference()
        for index in range(self.biases[-1].shape[0]):
            self.biases[-1][index] += self.alpha * network_error

    def input_hidden_weights_learning(self):
        hidden_layer_error = self.get_hidden_layer_error()
        for input_index in range(self.weights[0].shape[0]):
            current_input_value = self.values[0][0][input_index]
            for hidden_index in range(self.weights[0].shape[1]):
                self.weights[0][input_index][hidden_index] -= self.alpha * \
                                                              hidden_layer_error[hidden_index] * \
                                                              self.reverse_context_activation_function() * \
                                                              current_input_value

    def context_hidden_weights_learning(self):
        hidden_layer_error = self.get_hidden_layer_error()
        for weight_index in range(self.context_weights.shape[0]):
            self.context_weights[weight_index] -= self.alpha * \
                                                  hidden_layer_error[weight_index] * \
                                                  self.reverse_context_activation_function() * \
                                                  self.context_value

    def hidden_biases_learning(self):
        hidden_layer_error = self.get_hidden_layer_error()
        for bias_index in range(self.biases[0].shape[0]):
            self.biases[0][bias_index] += self.alpha * \
                                          hidden_layer_error[bias_index] * \
                                          self.reverse_context_activation_function()

    def learning(self, input_data, excellent_example, expected_error):
        self.excellent_example = excellent_example
        steps = 0
        while (len(self.values) == 0 or self.get_network_error() > expected_error) and \
                steps < self.steps_porder:
            self.process(input_data)
            self.hidden_output_weights_learning()
            self.output_biases_learning()
            self.input_hidden_weights_learning()
            self.context_hidden_weights_learning()
            self.hidden_biases_learning()
            steps += 1

    @staticmethod
    def context_activation_function(argument):
        return asinh(argument)

    @staticmethod
    def reverse_context_activation_function():
        return 1


if __name__ == "__main__":
    pass
