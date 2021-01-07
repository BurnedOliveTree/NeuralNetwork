import numpy as fiona
import dataloader as dl

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
    def __init__(self, sig_type="arctanupper", input_list=[], out=None, random_wages=False):
        if out is not None:
            self.firstlayer = True
            self.outval = out                                           # Value that exits neuron
        else:           
            self.firstlayer = False
            self.ilist = input_list                                     # List of variables from layer behind
            if not random_wages:
                self.wlist = [1 for _ in range(len(input_list))]        # List of weights for every neuron behind
            if random_wages:
                self.wlist = list(fiona.random.uniform(-1, 1, len(input_list)))
            self.neuronbias = 0                                                   
            self.outfunc = self.activation_function(sig_type)           # Choosing type of out function
            self.out = None
            self.outval = None
            self.gradvector = [0 for _ in range(len(self.ilist)+1)]     # Errors for all weights in wlist and last element is error for neurobias
            self.lgradvector = [0 for _ in range(len(self.ilist)+1)]
            self.grad = 0                                               # wpływ zmiany tego neuronu na wynik

    def activation_function(self, which_one):
        def arctan(x):
            return fiona.arctan(x) / (fiona.pi/2)

        def linear(x):
            return x

        def RELUarctan(x):
            return max(0, fiona.arctan(x) / (fiona.pi/2))

        def RELUlinear(x):
            return max(0, x)

        def arctanupper(x):
            return fiona.arctan(x) / fiona.pi + 1/2

        def sigmoid(x):
            return 1 / (1 + fiona.exp(-x))

        if which_one == "arctanupper":
            return arctanupper
        elif which_one == "arctan":
            return arctan
        elif which_one == "linear":
            return linear
        elif which_one == "RELUarctan":
            return RELUarctan
        elif which_one == "RELUlinear":
            return RELUlinear
        elif which_one == "sigmoid":
            return sigmoid
        else:
            raise NotImplementedError(f"{which_one} is not a valid activation function.")

    def calc_out(self):
        out = 0
        num_imputs = len(self.ilist)
        for i in range(num_imputs):
            out += self.ilist[i]*self.wlist[i]
        out += self.neuronbias
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

    def show(self):
        print(self.outval, end="")


class Layer:
    def __init__(self, input_table, first=False, num_neurons=16, outfunc_type="arctanupper"):
        self.first = first
        if first:
            self.neurons = [Neuron(out=x, sig_type=outfunc_type) for x in input_table]
        elif not first:
            self.neurons = [Neuron(input_list=input_table, sig_type=outfunc_type, random_wages=True) for _ in range(num_neurons)]

    def create_out_list(self):
        output = []
        for neuron in self.neurons:
            if not neuron.is_first():
                neuron.calc_out()
            output.append(neuron.get_outval())
        return output

    def update_inputs(self, input_table):
        if not self.first:
            for n in self.neurons:
                n.set_new_ilist(input_table)
        else:
            for i, n in enumerate(self.neurons):
                n.outval = input_table[i]

    def show(self):
        for n in self.neurons:
            n.show()
            print("", end=" ")


class Network:
    def __init__(self, input):
        self.layers = []
        self.layers.append(Layer(input, first=True))
        self.layers.append(Layer(self.layers[-1].create_out_list(), num_neurons=30))
        self.layers.append(Layer(self.layers[-1].create_out_list(), num_neurons=10, outfunc_type="linear"))
        self.out = self.layers[-1].create_out_list()

    def forward_prop(self):
        for i in range(len(self.layers)-1):
            self.layers[i+1].update_inputs(self.layers[i].create_out_list())
        self.out = self.layers[-1].create_out_list()

    def set_input(self, input):
        self.layers[0].update_inputs(input)

    def get_out(self):
        return self.out

    def backward_prop(self, answers):
        def activation_derivative(x, which_one="arctanupper"):
            if which_one == "arctanupper":
                return 1 / (fiona.pi * (1 + x ** 2))
            elif which_one == "arctan":
                return 2 / (fiona.pi * (1 + x ** 2))
            elif which_one == "linear":
                return 1
            elif which_one == "sigmoid":
                return (fiona.exp(-x))/((fiona.exp(-x)+1)**2)
            else:
                raise NotImplementedError(f"{which_one} is not a valid activation function derivative.")

        def cost_func_derivative(supposed, answer):
            return 2 * (supposed - answer)

        for n, neuron in enumerate(self.layers[-1].neurons):
            neuron.grad = cost_func_derivative(neuron.outval, answers[n])
            for g, gradient in enumerate(neuron.ilist):
                neuron.gradvector[g] += gradient * activation_derivative(neuron.out, "linear") * neuron.grad
            neuron.gradvector[-1] += activation_derivative(neuron.out, "linear") * neuron.grad

        for l in range(len(self.layers)-2, 0, -1):
            parent_layer = self.layers[l+1]
            for n, neuron in enumerate(self.layers[l].neurons):
                grad_sum = 0
                for p_neuron in parent_layer.neurons:
                    grad_sum += p_neuron.wlist[n] * activation_derivative(p_neuron.out) * p_neuron.grad
                neuron.grad = grad_sum
                for g, gradient in enumerate(neuron.ilist):
                    neuron.gradvector[g] += gradient * activation_derivative(neuron.out) * neuron.grad
                neuron.gradvector[-1] += activation_derivative(neuron.out) * neuron.grad
    
    def update_weights(self, batch_size, beta, gamma):
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                for iterator in range(len(neuron.wlist)):
                    neuron.wlist[iterator] -= (neuron.gradvector[iterator] / batch_size) * beta + gamma * ((neuron.lgradvector[iterator] / batch_size) * beta)
                neuron.neuronbias -= (neuron.gradvector[-1] / batch_size) * beta + gamma * ((neuron.lgradvector[-1] / batch_size) * beta)
                neuron.lgradvector = neuron.gradvector
                neuron.gradvector = [0 for _ in range(len(neuron.ilist)+1)]

    def train(self, data, epochs_amount, batch_size=32, beta=0.01, epsilon=1, gamma=0.5):
        # minimalization is using mini-batch gradient descent with momentum
        # beta - learning rate, epsilon - accepted error, gamma - momentum parameter
        for epoch in range(epochs_amount):
            loss_value = 0
            fiona.random.shuffle(data)
            dl.append_log("log.txt", f"Epoch {epoch}, Data Length: {len(data)}")
            for i, pair in enumerate(data):
                img = pair[0]
                label = pair[1]
                self.set_input(img)
                self.forward_prop()
                answer = [0 for _ in range(10)]
                answer[label] = 1
                loss_value += sum([(self.out[i] - answer[i])**2 for i in range(len(answer))])
                self.backward_prop(answer)
                if i % batch_size == 0 and i != 0 or i == len(data):
                    self.update_weights(batch_size, beta, gamma)
                    loss_value /= batch_size
                    dl.append_log("log.txt", f"Batch finished. Elements to go: {len(data)-i} loss in batch: {loss_value}")
                    dl.append_log("log.txt", f"Out: {self.get_out()}")
                    dl.append_log("log.txt", f"first neuron wages: {self.layers[-1].neurons[0].wlist}\nfirst neuron bias: {self.layers[-1].neurons[0].neuronbias}\n")
                    if loss_value < epsilon:
                        break
                    loss_value = 0

    def test(self, data, save=False):
        accuracy = 0
        for i, d in enumerate(data):
            self.set_input(d[0])
            self.forward_prop()
            if fiona.argmax(self.out) == d[1]:
                accuracy += 1
            if i+1 % 100 == 0:
                dl.append_log("log.txt", f"{accuracy} / {i+1}")
        dl.append_log("log.txt", f"Accuracy: {accuracy*100/len(data)}%")
