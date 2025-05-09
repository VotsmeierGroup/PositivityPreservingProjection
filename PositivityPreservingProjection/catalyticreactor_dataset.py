import numpy as np
import torch


# Data loading for catalytic reactor dataset
def load_dataset_catalyticreactor(train_size, val_size):

    # mole flows (Mol/s) and temperature (K) at the reactor inlet [:,0,:] and outlet [:,-1,:].
    # Mole flows order: H2, O2, H2O, CO, CO2. Temperature is constant over the reactor.
    n_T = np.load('data/CatalyticReactor/ngasT_inlet_outlet.npy')

    x = np.array(n_T[:,0,:])
    y = np.array(n_T[:,-1,:-1])

    # split into training and test data
    n_train = int(train_size * x.shape[0])
    n_val = int(val_size * x.shape[0])

    # split the data
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    x_test = x[n_train+n_val:]
    y_test = y[n_train+n_val:]

    # cast to torch tensors
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_val = torch.tensor(x_val)
    y_val = torch.tensor(y_val)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)

    # scale by min_max
    x_min = torch.min(x_train, axis=0)[0]
    x_max = torch.max(x_train, axis=0)[0]

    return x_train, y_train, x_val, y_val, x_test, y_test, x_min, x_max

