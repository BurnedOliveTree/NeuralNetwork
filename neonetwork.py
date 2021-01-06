import math as shrek
from os import truncate
import numpy as fiona
import random
from copy import deepcopy
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
    def __init__(self, sig_type="arctanupper", input_list = [], out = None):
        if out != None:
            self.firstlayer = True
            self.outval = out                                                   # Value that exits neuron
        else:           
            self.firstlayer = False
            self.ilist = input_list                                             # List of variables from layer behind
            self.wlist = [1 for x in range(len(input_list))]                    # List of weights for every neuron behind
            self.neuronbias = 0                                                   
            self.outfunc = self.sigmoid(sig_type)                               # Chosing type of out function
            self.out = None
            self.outval = None
            self.error = 0
            self.gradvector = [0 for x in range(len(self.ilist)+1)]             # Errors for all weights in wlist and last element is error for neurobias 
            self.grad = 0                                                       # wpływ zmiany tego neuronu na wynik

    def sigmoid(self, which_one):
        def arctan(x):
            return shrek.atan(x)/(shrek.pi/2)
        def linear(x):
            return x
        def RELUarctan(x):
            return max(0, shrek.atan(x)/(shrek.pi/2))
        def RELUlinear(x):
            return max(0, x)
        def arctanupper(x):
            return shrek.atan(x) / shrek.pi +1/2
        if which_one == "arctanupper":
            return arctanupper
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
            out += self.ilist[i]*self.wlist[i]
        out+=self.neuronbias
        self.out = out
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

    '''
    def backward_prop_error(self, labels):
        def sigmoid_derivative(x):
            # derivative of 1 / (1 + e^(-x)):  x * (1 - x)
            # derivative of arctan:
            return fiona.pi / (2 * (1 + x ** 2))

        # last layer error calculations
        for i, neuron in enumerate(self.layers[-1].neurons):
            error = labels[i] - neuron.outval
            neuron.error = error * sigmoid_derivative(neuron.outval)

        # other layer calculations starting from second to last
        for i in reversed(range(1, len(self.layers) - 1)):
            for j, neuron in enumerate(self.layers[i].neurons):
                error = sum([neuron.wlist[j] * neuron_1.error for neuron_1 in self.layers[i + 1].neurons])
                neuron.error = error * sigmoid_derivative(neuron.outval)
    '''

    def backward_prop(self, answers):
        def sigmoid_derivative(x):
            # derivative of arctan * pi/2:
            #return 2 / (fiona.pi * (1 + x ** 2))
            # derivative of arctan * 1/pi + 1/2:
            return 1 / (fiona.pi * (1 + x ** 2))

        def cost_func_derivative(supposed, answer):
            return 2*(supposed - answer)

        for iterator, neuron in enumerate(self.layers[-1].neurons):
            neuron.grad = cost_func_derivative(neuron.outval, answers[iterator])
            for graditer in range(len(neuron.ilist)):
                neuron.gradvector[graditer] += neuron.ilist[graditer] * sigmoid_derivative(neuron.out) * neuron.grad
            neuron.gradvector[len(neuron.ilist)] += sigmoid_derivative(neuron.out) * neuron.grad
            neuron.grad = cost_func_derivative(neuron.outval, answers[iterator])

        for iterator in range(len(self.layers)-2, 0, -1):
            parentlayer = self.layers[iterator+1]
            layer = self.layers[iterator]
            for neuroniter, neuron in enumerate(layer.neurons):
                grad_avg = 0
                for pneuron in parentlayer.neurons:
                    grad_avg += pneuron.wlist[neuroniter] * sigmoid_derivative(pneuron.out) * pneuron.grad
                grad_avg/=len(parentlayer.neurons)
                neuron.grad = grad_avg
                for graditer in range(len(neuron.ilist)):
                    neuron.gradvector[graditer] += neuron.ilist[graditer] * sigmoid_derivative(neuron.out) * neuron.grad
                neuron.gradvector[len(neuron.ilist)] += sigmoid_derivative(neuron.out) * neuron.grad

    '''
    def update_weights_error(self, image, l_rate):
        for i in range(1, len(self.layers)):
            inputs = [neuron.outval for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(inputs)):
                    neuron.wlist[j] += l_rate * neuron.error * inputs[j]
                neuron.neuronbias += l_rate * neuron.error
    '''
    
    def update_weights(self, iterations_in_batch, lrate):
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                for iterator in range(len(neuron.wlist)):
                    neuron.wlist[iterator] -= (neuron.gradvector[iterator] / iterations_in_batch) * lrate 
                    neuron.gradvector[iterator] = 0
                neuron.neuronbias -= neuron.gradvector[-1]
                neuron.gradvector[len(neuron.wlist)] = 0
 
    """
    def train(self, data, all_labels, max_iterations, batch_size=64, epsilon=0.01):
        assert len(data) == len(all_labels)
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
                labels[all_labels[indexes[label]]] = 1
                loss_value += sum([(self.out[i] - labels[i])**2 for i in range(len(labels))])
                self.backward_prop_error(labels)
                self.update_weights_error(image, l_rate=0.1)
            print(f'Loss value equals {loss_value} in epoch {epoch}')
            if abs(loss_value - old_loss_value) < epsilon:
                return
    """

    def train(self, data, max_iterations, batch_size=32, epsilon=35, lrate = 10):
        end_all = False
        for epoch in range(max_iterations):
            end_epoch = False
            bsize = 64
            loss_value = 0
            epoch_data = deepcopy(data)
            print(f"Epoch Data Length: {len(epoch_data)}")
            while True: #pls do not hit me
                batch = []
                loss_value = 0
                for iter in range(batch_size):
                    index = random.randint(0,len(epoch_data)-1)
                    batch.append(epoch_data[index])
                    epoch_data.pop(index)
                    if len(epoch_data)==0:
                        end_epoch= True
                        bsize = iter
                        break
                for pair in batch:
                    img = pair[0]
                    label = pair[1]
                    self.set_input(img)
                    self.forward_prop()
                    answer = [0 for x in range(10)]
                    answer[label] = 1
                    loss_value += sum([(self.out[i] - answer[i])**2 for i in range(len(answer))])
                    self.backward_prop(answer)
                self.update_weights(bsize, lrate)
                print(f"Batch finished. Elements to go: {len(epoch_data)} loss in batch: {loss_value/bsize}")
                if end_epoch:
                    print(f"EPOCH {epoch}")
                    break
            if loss_value<epsilon:
                break

    def test(self, data):
        how_good = 0
        for it, d in enumerate(data):
            self.set_input(d[0])
            self.forward_prop()
            answers = self.out
            maxind = 0
            maxval = 0
            for i, v in enumerate(answers):
                if v > maxval:
                    maxind = i
                    maxval = v
            if maxind == d[1]:
                how_good+=1
            print(f"{how_good}/{it}")


'''
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
'''