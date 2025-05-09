import torch

torch.set_default_dtype(torch.double)


class baseNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, data_min, data_max):
        super(baseNeuralNetwork, self).__init__()
        
        # Defining hyperparameters
        self.input_size = input_size  # Size of the input layer
        self.hidden_size  = hidden_size  # Size of the hidden layers
        self.output_size = output_size  # Size of the output layer
        self.num_layers = num_layers  # Number of hidden layers
        self.layers = torch.nn.ModuleList()  # We use a ModuleList to store the layers of the network
        self.data_min = data_min
        self.data_max = data_max

        # Input layer
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size)) 

        # Hidden layers
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            
        # output layer
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))
    
    def scale_inputs(self, x):
        # We scale the input data to the range [-1, 1]
        return 2*(x - self.data_min) / (self.data_max - self.data_min) - 1


class expNeuralNetwork(baseNeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, num_layers, data_min, data_max):
        super(expNeuralNetwork, self).__init__(input_size, hidden_size, output_size, num_layers, data_min, data_max)

    def forward(self, x):

        # We scale the input data to the range [-1, 1]
        x = self.scale_inputs(x)

        # We loop through the layers of the network and apply the tanh activation function
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)
        
        # Output layer with exponential activation function
        y = torch.exp(self.layers[-1](x))
        
        return y 


class standardNeuralNetwork(baseNeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, num_layers, data_min, data_max):
        super(standardNeuralNetwork, self).__init__(input_size, hidden_size, output_size, num_layers, data_min, data_max)

    def forward(self, x):

        # We scale the input data to the range [-1, 1]
        x = self.scale_inputs(x)

        # We loop through the layers of the network and apply the tanh activation function
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)
        
        # Output layer with no activation function
        y = self.layers[-1](x)
        
        return y

