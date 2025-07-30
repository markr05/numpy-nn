import numpy as np
import time

start_time = time.time()
data = np.loadtxt("number-set/mnist_test.csv", delimiter=",").astype(np.float32)
labels = np.array(data[:, 0])

correct_answers = np.eye(10, dtype=np.float32)[labels.astype(int)]
images = np.array(data[:, 1:] / 255, dtype=np.float32)

data2 = np.loadtxt("number-set/mnist_train.csv", delimiter=",")
labels2 = np.array(data2[:, 0])

Y_Train = np.eye(10, dtype=np.float32)[labels2.astype(int)]
X_Train = np.array(data2[:, 1:] / 255, dtype=np.float32)

class NeuralNetwork:
  def __init__(self, data, answers, layers=1, num_neurons=128, num_answer_choices=10, weights=None, biases=None, learning_rate=0.55):
    self.data = data
    self.data_size = len(self.data)
    self.num_answer_choices = num_answer_choices
    self.correct_answers = answers
    self.num_hidden_layers = layers
    self.num_neurons = num_neurons
    self.input_size = len(self.data[0])
    self.learning_rate = learning_rate

    if weights == None:
      self.weights = []
      weight_range = np.sqrt(6.0 / self.input_size)
      self.weights.append(np.random.uniform(-weight_range, weight_range, (self.input_size, self.num_neurons)).astype(np.float32))
      for _ in range(self.num_hidden_layers):
        weight_range = np.sqrt(6.0 / (self.num_neurons + self.num_neurons))
        self.weights.append(np.random.uniform(-weight_range, weight_range, (self.num_neurons, self.num_neurons)).astype(np.float32))
      weight_range = np.sqrt(6.0 / (self.num_neurons + self.num_answer_choices))
      self.weights.append(np.random.uniform(-weight_range, weight_range, (self.num_neurons, self.num_answer_choices)).astype(np.float32))
    else:
      self.weights = weights

    if biases == None:
      self.biases = []
      for _ in range(self.num_hidden_layers + 1):
        self.biases.append(np.zeros(self.num_neurons).astype(np.float32))
      self.biases.append(np.zeros(self.num_answer_choices).astype(np.float32))
    else:
      self.biases = biases
  
  def leaky_relu(self, inp, weights, biases, alpha=0.01):
    output = (inp @ weights + biases).astype(np.float32)
    return ((output), np.where(output >= 0, output, alpha * output))

  def softmax(self):
    z_shifted = (self.z_values[-1] - np.max(self.z_values[-1], axis=1, keepdims=True))
    exp_z = np.exp(z_shifted).astype(np.float32)
    self.selections = (exp_z / np.sum(exp_z, axis=1, keepdims=True))
  
  def leaky_relu_derivative(self, z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
  
  def backprop_eq1(self):
    return (self.selections - self.correct_answers)
  
  def backprop_eq2(self, layer):
    raw_gradient = self.error_values[layer + 1] @ self.weights[layer + 1].T
    scaled_gradient = raw_gradient / self.num_neurons
    return scaled_gradient * self.leaky_relu_derivative(self.z_values[layer])
  
  def forward_propagation(self):
    x = self.data
    self.z_values = []
    self.activations = []

    for i in range(self.num_hidden_layers + 1):
      layer_z, layer_activation = self.leaky_relu(x, self.weights[i], self.biases[i])
      self.z_values.append(layer_z)
      self.activations.append(layer_activation)
    
      x = layer_activation
    final_z = (x @ self.weights[-1] + self.biases[-1])
    self.z_values.append(final_z)

    self.softmax()
    self.activations.append(self.selections)

  def backpropagation(self):
    # error calculations for the last layer
    self.error_values = []
    for _ in range(self.num_hidden_layers + 1):
      self.error_values.append(np.zeros(shape=(self.data_size, self.num_neurons)))
    self.error_values.append(np.zeros(shape=(self.data_size, self.num_answer_choices)))

    self.error_values[-1] = self.backprop_eq1()
    # error calculations for the rest of the layers
    for layer in range(self.num_hidden_layers, -1, -1):
      self.error_values[layer] = self.backprop_eq2(layer)

      bias_gradient = np.mean(self.error_values[layer], axis=0)
      self.biases[layer] -= self.learning_rate * bias_gradient
      
      input_activations = self.activations[layer - 1] if layer > 0 else self.data
      weight_gradient = input_activations.T @ self.error_values[layer]
      self.weights[layer] -= self.learning_rate * weight_gradient

# Testing
batch_size = 128
num_epochs = 60  # or however many you want

mnist_nn = NeuralNetwork(X_Train[:batch_size], Y_Train[:batch_size])
for epoch in range(num_epochs):
  permutation = np.random.permutation(X_Train.shape[0])
  X_shuffled = X_Train[permutation]
  y_shuffled = Y_Train[permutation]

  for i in range(0, X_Train.shape[0], batch_size):
    batch_x = X_shuffled[i:i+batch_size]
    batch_y = y_shuffled[i:i+batch_size]

    mnist_nn.data = batch_x
    mnist_nn.correct_answers = batch_y
    mnist_nn.data_size = len(batch_x)

    mnist_nn.forward_propagation()
    mnist_nn.backpropagation()
    
  mnist_nn.data = X_Train
  mnist_nn.correct_answers = Y_Train
  mnist_nn.data_size = len(X_Train)
  mnist_nn.forward_propagation()
  acc = np.mean(np.argmax(mnist_nn.selections, axis=1) == np.argmax(Y_Train, axis=1)) * 100
  print(f"Epoch {epoch}: Accuracy = {acc:.2f}%")
full_time = time.time() - start_time
print(f"Time taken: {full_time:.2f} seconds")

# Testing on the test set
mnist_test = NeuralNetwork(images, correct_answers, weights=mnist_nn.weights, biases=mnist_nn.biases)

mnist_test.forward_propagation()
total_correct = np.sum(np.argmax(mnist_test.selections, axis=1) == np.argmax(mnist_test.correct_answers, axis=1))
print(f"Total Correct: {total_correct}/{mnist_test.data_size}")
