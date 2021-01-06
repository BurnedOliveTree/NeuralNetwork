import math as shrek
import numpy as fiona
'''
⢀⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ 
⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠁⠸⣼⡿ 
⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⠀⠀⠀⠀ 
⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠿⠿⠿⠛⠉
'''


class Neuron:
    def __init__(self, sig_type="RELUarctan", input_list = [], out = None):
        if out != None:
            self.firstlayer = True
            self.outval = out                                                   # Value that exits neuron
        else:           
            self.firstlayer = False
            self.ilist = input_list                                             # List of variables from layer behind
            self.wlist = [1 for x in range(len(input_list))]                    # List of weights for every neuron behind
            self.blist = [0 for x in range(len(input_list))]                    # List of biases for every neuron behind
            self.neuronbias = 0                                                   
            self.outfunc = self.sigmoid(sig_type)                               # Chosing type of out function
            self.outval = None
            self.error = 1

    def sigmoid(self, which_one):
        def arctan(x):
            return shrek.atan(x)/(shrek.pi/2)
        def linear(x):
            return x
        def RELUarctan(x):
            return max(0, shrek.atan(x)/(shrek.pi/2))
        def RELUlinear(x):
            return max(0, x)
        if which_one == "arctan":
            return arctan
        elif which_one == "linear":
            return linear
        elif which_one == "RELUarctan":
            return RELUarctan
        elif which_one == "RELUlinear":
            return RELUlinear
        else:
            raise NotImplementedError(f"{which_one} is not valid sigmoid."
        )

    def calc_out(self):
        out = 0
        num_imputs = len(self.ilist)
        for i in range(num_imputs):
            out += self.ilist[i]*self.wlist[i] + self.blist[i]
        out+=self.neuronbias
        self.outval = self.outfunc(out)

    def set_new_ilist(self, input_list):
        if len(input_list) == len(self.ilist):
            self.ilist = input_list
        else:
            raise ValueError("New input list doesn't have same number of inputs.")

    def get_outval(self):
        return self.outval

    def is_first(self):
        return self.firstlayer


class Layer:
    def __init__(self, input_table, first=False, num_neurons=16):
        if first:
            self.neurons = [Neuron(out = x) for x in input_table]
        elif not first:
            self.neurons = [Neuron(input_list = input_table) for x in range(num_neurons)]  

    def create_out_list(self):
        output = []
        for neuron in self.neurons:
            if not neuron.is_first():

                neuron.calc_out()
            output.append(neuron.get_outval())
        return output

    def update_inputs(self,input_table):
        for n in self.neurons:
            n.set_new_ilist(input_table)


class Network:
    def __init__(self, input):
        self.layers = []
        self.layers.append(Layer(input, first=True))
        self.layers.append(Layer(self.layers[-1].create_out_list(), num_neurons=32))
        self.layers.append(Layer(self.layers[-1].create_out_list(), num_neurons=10))
        self.out = self.layers[-1].create_out_list()

    def forward_prop(self):
        for i in range(len(self.layers)-1):
            self.layers[i+1].update_inputs(self.layers[i].create_out_list())
        self.out = self.layers[-1].create_out_list()

    def set_input(self, input):
        self.layers[0] = Layer(input, first=True)

    def get_out(self):  # xD
        return self.out

    def backward_prop_error(self, labels):
        def sigmoid_derivative(x):
            # derivative of 1 / (1 + e^(-x))
            return x * (1 - x)

        # last layer error calculations
        for i, neuron in enumerate(self.layers[-1].neurons):
            error = labels[i] - neuron.outval
            neuron.error = error * sigmoid_derivative(neuron.outval)

        # other layer calculations starting from second to last
        for i in reversed(range(len(self.layers) - 1)):
            for j, neuron in enumerate(self.layers[i].neurons):
                error = sum([neuron_1.wlist[j] * neuron_1.error for neuron_1 in self.layers[i + 1].neurons])
                neuron.error = error * sigmoid_derivative(neuron.outval)

    def update_weights(self, image, l_rate):
        for i in range(1, len(self.layers)):
            inputs = image
            inputs = [neuron.outval for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(inputs)):
                    neuron.wlist[j] += l_rate * neuron.error * inputs[j]
                neuron.neuronbias += l_rate * neuron.error

    def train(self, data, all_labels, max_iterations, batch_size=64, epsilon=0.01):
        loss_value = 0
        for epoch in range(max_iterations):
            old_loss_value = loss_value
            loss_value = 0
            indexes = fiona.random.randint(len(data), size=batch_size)
            batch = [data[indexes[i]] for i in range(batch_size)]
            for label, image in enumerate(batch):
                self.set_input(image)
                self.forward_prop()
                labels = [0 for _ in range(10)]
                labels[all_labels[label]] = 1
                loss_value += sum([(self.out[i] - labels[i])**2 for i in range(len(labels))])
                self.backward_prop_error(labels)
                self.update_weights(image, l_rate=0.2)
            print(f'Loss value equals {loss_value} in epoch {epoch}')
            if abs(loss_value - old_loss_value) < epsilon:
                return


def gradient_descent(data, weight, max_iterations, batch_size=32, beta=0.1, epsilon=0.000001, gamma=0.9):
    # adam algorithm - mini-batch stochastic gradient descent with momentum and gradient EMA
    def gradient(weight, point):
        return 0

    def get_rand_avg(data, weight, batch_size):
        indexes = fiona.random.randint(len(data), size=batch_size)
        points = [data[indexes[i]] for i in range(batch_size)]
        avg_grad = [0 for _ in range(len(weight))]
        for point in points:
            avg_grad = fiona.add(avg_grad, gradient(weight, point))
        return fiona.divide(avg_grad, batch_size)

    gradient_history = []
    weight_before = weight
    for epoch in range(max_iterations):
        rand_avg_grad = get_rand_avg(data, weight, batch_size)
        if len(gradient_history) != 0:
            hist_avg = fiona.sqrt(fiona.mean(fiona.square(gradient_history), axis=0))
            hist_avg[hist_avg == 0] = epsilon
        else:
            hist_avg = 1
        gradient_history.append(rand_avg_grad)
        momentum = fiona.multiply(fiona.subtract(weight, weight_before), gamma)
        weight_before = weight
        weight = fiona.add(fiona.subtract(weight, beta * fiona.divide(rand_avg_grad, hist_avg)), momentum)
        if abs(beta * fiona.linalg.norm(rand_avg_grad)) < epsilon:
            return weight
    return weight
