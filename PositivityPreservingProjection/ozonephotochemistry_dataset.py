import numpy as np
import pandas as pd


#To use this dataset, download the data from https://zenodo.org/records/13385987 and place it in the data folder

# Data loading for ozone photochemistry dataset, taken and adapted from https://zenodo.org/records/13385987


def load_dataset_ozonephotochemistry():
    n_points_per_experiment = 13  # 1 hours * 1 step/5 minutes + 1 for the first step
    n_steps_per_experiment = n_points_per_experiment - 1

    print("loading concentration data")
    # read dataframe from csv
    df = pd.read_csv('data/OzonePhotochemistry/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv')

    # Get concentration values C, which are all columns except the first two
    C = df.iloc[:, 2:].values

    # Get tendency values D, which are the difference between consecutive concentration values
    D = np.delete(np.diff(C, axis=0), list(range(n_steps_per_experiment, C.shape[0] - 1, n_points_per_experiment)),
                  axis=0)

    # Get C of active species, ignoring H2O, O2, and buildup HNO3, CO, H2
    ignore_species = ['H2O', 'O2', 'HNO3', 'CO', 'H2']
    active_species_columns = [col for col in df.columns if
                              col.split(' ')[0] not in ignore_species and col.endswith('[ppb]')]
    C_active = df[active_species_columns].values

    print("creating input and output data")
    # delete the last step of each experiment -> Because there is no assosciated delC for the last value of the
    # experiment. 13 steps = 12 delC values for delC = model(C0)
    X_all = np.delete(C,
                      list(range(n_steps_per_experiment,
                                 C_active.shape[0], n_points_per_experiment)),
                      axis=0)

    # Create a train/test split
    split = 0.90
    trainsplit = int(split * X_all.shape[0])
    print("train size:", trainsplit)
    testsplit = int(round(1 - split, 2) * X_all.shape[0])
    print("test size:", testsplit)
    num_test_exps = int(testsplit / (n_steps_per_experiment))
    print("number of test experiments:", num_test_exps)

    X_all_test = X_all[trainsplit:, :]
    Y_test = D[trainsplit:, :]

    return X_all_test, Y_test


# get active species, according to sturm. ignore_species = ['H2O', 'O2', 'HNO3', 'CO', 'H2']. These species are not
# used as predictors for the ML models
def get_activespecies(delC):
    delC_active = delC[:, [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13]]

    return delC_active

