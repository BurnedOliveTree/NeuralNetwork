import math as shrek
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
        self.layer0 = Layer(input, first=True)
        self.layer1 = Layer(self.layer0.create_out_list())
        self.layer2 = Layer(self.layer1.create_out_list(), num_neurons=10)
        self.out = self.layer2.create_out_list()

    def go_through(self):
        self.layer2.update_inputs(self.layer1.create_out_list())
        self.out = self.layer2.create_out_list()

    def get_out(self): #xD
        return self.out



        