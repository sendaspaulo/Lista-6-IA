import numpy as np
#calcula os valores de entrada para 0 e 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#calcula a derivada da sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)
#calcula a funcao de ativacao ReLU - retorna 0 se for negativa e o valor de entrada se for positivo
def relu(x):
    return np.maximum(0, x)
#calcula a derivada da funcao ReLU
def relu_derivative(x):
    return np.where(x <= 0, 0, 1)
#responsavel por gerar os valores de entrada para as portas lógicas.
#gera todas as entradas possiveis e calcula as saidas correspoondentes.
def generate_data(function, n_inputs):
    #gerando todas as combinacoes
    inputs = np.array(np.meshgrid(*[[0, 1]] * n_inputs)).T.reshape(-1, n_inputs)
    if function == "AND":
        outputs = np.all(inputs, axis=1).astype(int)
    elif function == "OR":
        outputs = np.any(inputs, axis=1).astype(int)
    elif function == "XOR":
        outputs = np.bitwise_xor.reduce(inputs, axis=1)
    return inputs, outputs.reshape(-1, 1)

class NeuralNetwork:
    #define pesos e Bias aleatorios
    def __init__(self, n_inputs, activation='sigmoid'):
        self.n_inputs = n_inputs
        self.hidden_neurons = 4  # numero de neuronios na camada oculta
        self.output_neurons = 1  # numero de neuronios na saida

        # inicializa pesos e bias
        self.weights_hidden = np.random.uniform(-1, 1, (self.n_inputs, self.hidden_neurons))
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_neurons))
        self.weights_output = np.random.uniform(-1, 1, (self.hidden_neurons, self.output_neurons))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_neurons))

        # funcao de ativacao
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
    #calcula as saidas dos neuronios nas camadas ocultas e de saidas
    def forward_pass(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.activation(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        return sigmoid(self.output_layer_input)
    #calcula o erro de saida e ajusta os pesos e bias pelo gradiente descendente
    def backward_pass(self, inputs, outputs, predicted, learning_rate):
        #calcula o erro de saida
        error_output = outputs - predicted
        delta_output = error_output * sigmoid_derivative(predicted)

        # erro da camada oculta
        error_hidden_layer = delta_output.dot(self.weights_output.T)
        delta_hidden_layer = error_hidden_layer * self.activation_derivative(self.hidden_layer_output)

        #atualiza os pesos e bias
        self.weights_output += self.hidden_layer_output.T.dot(delta_output) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_hidden += inputs.T.dot(delta_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learning_rate
    #treina a rede com os dados de entrada e saida fornecidos executando 'foward_pass' e 'backward_pass' por um numero determinado de vezes
    def train(self, inputs, outputs, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            predicted = self.forward_pass(inputs)
            self.backward_pass(inputs, outputs, predicted, learning_rate)

# experimentos com as portas logicas escolhidas
functions = ["AND", "OR", "XOR"]
n_inputs = 3  # numero variavel de entradas
results = []
#executa os experimentos para as funcoes logicas
for func in functions:
    inputs, outputs = generate_data(func, n_inputs)
    nn = NeuralNetwork(n_inputs, activation='sigmoid')
    nn.train(inputs, outputs, learning_rate=0.1, epochs=10000)
    predicted = nn.forward_pass(inputs)
    accuracy = np.mean(np.round(predicted) == outputs)
    results.append((func, accuracy))

print(results)
