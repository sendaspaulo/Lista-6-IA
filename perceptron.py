from functools import reduce
import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

def generate_data(operation, no_of_inputs):
    training_inputs = []
    labels = []
    for i in range(2**no_of_inputs):
        binary_input = [int(x) for x in format(i, '0' + str(no_of_inputs) + 'b')]
        training_inputs.append(np.array(binary_input))
        if operation == 'AND':
            labels.append(np.all(binary_input))
        elif operation == 'OR':
            labels.append(np.any(binary_input))
        elif operation == 'XOR':
            labels.append(reduce(lambda x, y: x ^ y, binary_input))
    return np.array(training_inputs), np.array(labels)

def test_operation(perceptron, operation, no_of_inputs):
    training_inputs, labels = generate_data(operation, no_of_inputs)
    perceptron.train(training_inputs, labels)
    print(f"\nTestando operação {operation} com {no_of_inputs} entradas:")
    for inputs, label in zip(training_inputs, labels):
        prediction = perceptron.predict(inputs)
        print(f"Input: {inputs} - Real: {label} - Predição: {prediction}")

# Testando AND e OR com diferentes números de entradas
no_of_inputs = [2, 10]  # Exemplos de números de entradas
for n in no_of_inputs:
    test_operation(Perceptron(n), 'AND', n)
    test_operation(Perceptron(n), 'OR', n)

# Demonstrando a incapacidade do Perceptron em resolver XOR
test_operation(Perceptron(2), 'XOR', 2)
