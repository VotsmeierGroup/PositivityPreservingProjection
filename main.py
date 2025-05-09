# generates results for catalytic reactor dataset, ozone photochemistry dataset, and random concentration dataset

# imports
import os
import torch
import yaml
import numpy as np
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from PositivityPreservingProjection.projections import orthogonal_projection, positivity_projection, orthogonal_backtracking, positivity_backtracking
from PositivityPreservingProjection.catalyticreactor_dataset import load_dataset_catalyticreactor
from PositivityPreservingProjection.ozonephotochemistry_dataset import load_dataset_ozonephotochemistry, get_activespecies
from PositivityPreservingProjection.randomconcentration_dataset import generate_c0_data, generate_c1_data
from PositivityPreservingProjection.models import expNeuralNetwork, standardNeuralNetwork
from PositivityPreservingProjection.errormetrics import calculate_error_metrics

EVALUATE_CATALYTIC_REACTOR = True
EVALUATE_OZONE_PHOTOCHEMISTRY = True   ## To use this dataset, download the data from https://zenodo.org/records/13385987 and place it in the data folder
EVALUATE_RANDOM_CONCENTRATION = True

if __name__ == '__main__':

    if not os.path.exists('results'):
        os.makedirs('results')

    if EVALUATE_CATALYTIC_REACTOR:

        # atom molecule matrix
        E = np.array([  [2.0, 0.0, 2.0, 0.0, 0.0],  # H
                        [0.0, 2.0, 1.0, 1.0, 2.0],  # O
                        [0.0, 0.0, 0.0, 1.0, 1.0]])  # C

        # load data
        x_train, y_train, x_val, y_val, x_test, y_test, x_min, x_max = load_dataset_catalyticreactor(train_size=0.4, val_size=0.1)

        # load model
        model_save = torch.load("models/CatalyticReactorModel.pt")
        # load config
        with open('models/CatalyticReactorModelConfig.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        x_min = model_save['data_min']
        x_max = model_save['data_max']
        model = expNeuralNetwork(input_size=x_train.shape[1], hidden_size=config["architecture_args"]["hidden_size"], output_size=y_train.shape[1], num_layers=config["architecture_args"]["num_layers"], data_min=x_min, data_max=x_max)
        model.load_state_dict(model_save['model_state_dict'])

        # generate predictions
        y_pred_test = model(x_test)
        c_in = x_test[:,:-1].detach().numpy()
        c_out_true = y_test.detach().numpy()
        model_pred = y_pred_test.detach().numpy()

        # generate error metrics
        c_out_corrected_positivity = positivity_projection(c_in, model_pred, E)
        c_out_corrected_orthogonal = orthogonal_projection(c_in, model_pred, E)

        # calculate error metrics
        atombalance_mean_err_positivity, neg_count_perc_positivity, rmse_positivity, rel_err_positivity = calculate_error_metrics(c_in, c_out_corrected_positivity, c_out_true, E)
        atombalance_mean_err_orthogonal, neg_count_perc_orthogonal, rmse_orthogonal, rel_err_orthogonal = calculate_error_metrics(c_in, c_out_corrected_orthogonal, c_out_true, E)
        atombalance_mean_err_model, neg_count_perc_model, rmse_model, rel_err_model = calculate_error_metrics(c_in, model_pred, c_out_true, E)

        summary = {
            'model': {
                'rmse': float(rmse_model),
                'mean_err_atombalance': float(atombalance_mean_err_model),
                'neg_count': float(neg_count_perc_model),
                'rel_err': float(rel_err_model)
            },
            'positivity_projection': {
                'rmse': float(rmse_positivity),
                'mean_err_atombalance': float(atombalance_mean_err_positivity),
                'neg_count': float(neg_count_perc_positivity),
                'rel_err': float(rel_err_positivity)
            },
            'orthogonal_projection': {
                'rmse': float(rmse_orthogonal),
                'mean_err_atombalance': float(atombalance_mean_err_orthogonal),
                'neg_count': float(neg_count_perc_orthogonal),
                'rel_err': float(rel_err_orthogonal)
            }
        }

        # Save the summary to a YAML file
        with open('results/CatalyticReactor_table.yaml', 'w') as file:
            # Dump the data into the file in YAML format
            yaml.dump(summary, file, default_flow_style=False)

    if EVALUATE_OZONE_PHOTOCHEMISTRY:

        # atom molecule matrix
        E = np.array([  [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 2., 3., 2., 2., 0., 0.],       # C
                        [0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],       # N
                        [0., 0., 0., 2., 1., 2., 1., 1., 0., 2., 4., 4., 3., 3., 2., 0.],       # H
                        [3., 1., 2., 1., 2., 2., 1., 3., 1., 0., 1., 2., 3., 5., 1., 2.]])      # O

        # Value to clip negative concentrations to
        CLIP_VALUE = 1e-4

        #load data
        test_C0_true, test_delC_true = load_dataset_ozonephotochemistry()
        test_C_next_true = test_C0_true + test_delC_true

        # load model
        model_save = torch.load("models/OzonePhotochemistryModel.pt")
        # load config
        with open('models/OzonePhotochemistryConfig.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        x_min = model_save['data_min']
        x_max = model_save['data_max']
        model = standardNeuralNetwork(input_size=get_activespecies(test_C0_true).shape[1], hidden_size=config["architecture_args"]["hidden_size"], output_size=test_delC_true.shape[1], num_layers=config["architecture_args"]["num_layers"], data_min=x_min, data_max=x_max)
        model.load_state_dict(model_save['model_state_dict'])

        # generate predictions and corrections
        delC_predict_NN = model(torch.tensor(get_activespecies(test_C0_true))).detach().numpy()  # not all species, only active species are used as predictors
        C_next_NN = test_C0_true + delC_predict_NN  # predicted concentrations
        C_next_NN_clip = np.maximum(C_next_NN, CLIP_VALUE)  # clip negative values to a small positive value
        C_next_NN_positivity = positivity_projection(test_C0_true, C_next_NN_clip, E)  # positivity projection
        C_next_NN_orthogonal = orthogonal_projection(test_C0_true, C_next_NN, E)  # orthogonal projection

        # generate error metrics
        atombalance_mean_err_model, neg_count_perc_model, rmse_model, _ = calculate_error_metrics(test_C0_true, C_next_NN, test_C_next_true, E)
        atombalance_mean_err_positivity, neg_count_perc_positivity, rmse_positivity, _ = calculate_error_metrics(test_C0_true, C_next_NN_positivity, test_C_next_true, E)
        atombalance_mean_err_orthogonal, neg_count_perc_orthogonal, rmse_orthogonal, _ = calculate_error_metrics(test_C0_true, C_next_NN_orthogonal, test_C_next_true, E)

        summary = {
            'model': {
                'atom_balance_mean_error': float(atombalance_mean_err_model),
                'negative_count_percentage': float(neg_count_perc_model),
                'rmse': float(rmse_model)
            },
            'positivity_projection': {
                'atom_balance_mean_error': float(atombalance_mean_err_positivity),
                'negative_count_percentage': float(neg_count_perc_positivity),
                'rmse': float(rmse_positivity)
            },
            'orthogonal_projection': {
                'atom_balance_mean_error': float(atombalance_mean_err_orthogonal),
                'negative_count_percentage': float(neg_count_perc_orthogonal),
                'rmse': float(rmse_orthogonal)
            }
        }

        # Save the summary to a YAML file
        with open('results/OzonePhotochemistry_table.yaml', 'w') as file:
            # Dump the data into the file in YAML format
            yaml.dump(summary, file, default_flow_style=False)
        
    if EVALUATE_RANDOM_CONCENTRATION:

        N_ATOM_MOLECULE_SYSTEMS = 100  # the paper shows results for 10,000 system. To keep runtime short for this script, we use 100 systems here
        N_CONDITIONS = 1000  # the paper shows results for 10,000 conditions. To keep runtime short for this script, we use 1000 conditions here
        NOISE_FACTORS_GAMMA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        all_rel_err = np.zeros((N_ATOM_MOLECULE_SYSTEMS, len(NOISE_FACTORS_GAMMA), 4))
        all_n_negative = np.zeros((N_ATOM_MOLECULE_SYSTEMS, len(NOISE_FACTORS_GAMMA), 4))
        all_atombalance_err = np.zeros((N_ATOM_MOLECULE_SYSTEMS, len(NOISE_FACTORS_GAMMA), 4))
        all_mean_noise_level = np.zeros((N_ATOM_MOLECULE_SYSTEMS, len(NOISE_FACTORS_GAMMA), 2))

        for atom_molecule_idx in range(N_ATOM_MOLECULE_SYSTEMS):

            sampled_E, c1_true = generate_c1_data(atom_molecule_idx, N_CONDITIONS)
            c0_true = generate_c0_data(atom_molecule_idx, sampled_E, c1_true)
            n_atoms, n_molecules = sampled_E.shape
            print(f"Atom-molecule pair {atom_molecule_idx + 1} out of {N_ATOM_MOLECULE_SYSTEMS}")

            rel_err = []
            n_negative = []
            atombalance_err = []
            mean_noise_level = []

            for gamma in NOISE_FACTORS_GAMMA:
                
                # make the data noisy
                beta = np.random.uniform(-gamma, gamma, (N_CONDITIONS, n_molecules))
                c1_noisy = c1_true * 10**beta

                # calculate the projection 
                cs_corrected_orthogonal = orthogonal_projection(c0_true, c1_noisy, sampled_E)
                cs_corrected_positivity = positivity_projection(c0_true, c1_noisy, sampled_E)
                cs_corrected_orthogonal_backtrack = orthogonal_backtracking(c0_true, c1_noisy, sampled_E, intersection_point=1e-10)
                cs_corrected_positivity_backtrack = positivity_backtracking(c0_true, c1_noisy, sampled_E, intersection_point=1e-10)

                # calculate the error metrics
                atom_balance_err_orthogonal, n_negative_orthogonal, _, rel_err_orthogonal = calculate_error_metrics(c0_true, cs_corrected_orthogonal, c1_true, sampled_E)
                atom_balance_err_positivity, n_negative_positivity, _, rel_err_positivity = calculate_error_metrics(c0_true, cs_corrected_positivity, c1_true, sampled_E)
                atom_balance_err_orthogonal_backtrack, n_negative_orthogonal_backtrack, _, rel_err_orthogonal_backtrack = calculate_error_metrics(c0_true, cs_corrected_orthogonal_backtrack, c1_true, sampled_E)
                atom_balance_err_positivity_backtrack, n_negative_positivity_backtrack, _, rel_err_positivity_backtrack = calculate_error_metrics(c0_true, cs_corrected_positivity_backtrack, c1_true, sampled_E)
                atom_balance_err_data, _, _, rel_err_data = calculate_error_metrics(c0_true, c1_noisy, c1_true, sampled_E)

                # save the error metrics
                rel_err.append([rel_err_orthogonal, rel_err_positivity, rel_err_orthogonal_backtrack, rel_err_positivity_backtrack])
                n_negative.append([n_negative_orthogonal, n_negative_positivity, n_negative_orthogonal_backtrack, n_negative_positivity_backtrack])
                atombalance_err.append([atom_balance_err_orthogonal, atom_balance_err_positivity, atom_balance_err_orthogonal_backtrack, atom_balance_err_positivity_backtrack])
                mean_noise_level.append([atom_balance_err_data, rel_err_data])
            
            all_rel_err[atom_molecule_idx] = rel_err
            all_n_negative[atom_molecule_idx] = n_negative
            all_atombalance_err[atom_molecule_idx] = atombalance_err
            all_mean_noise_level[atom_molecule_idx] = mean_noise_level

        # Find the lowest error at which the positivity projection yields a negative concentration
        all_first_negative = np.zeros(N_ATOM_MOLECULE_SYSTEMS)
        for idx in range(N_ATOM_MOLECULE_SYSTEMS):
            if np.any(all_n_negative[idx,:,1] > 0):
                first_negative = np.argmax(all_n_negative[idx,:,1]> 0)
            else:
                first_negative = len(NOISE_FACTORS_GAMMA)-1

            all_first_negative[idx] = first_negative

        # sort the arrays
        all_first_negative = np.sort(all_first_negative)
        print("Lowest error at which a negative concentration is obtained at gamma = ", NOISE_FACTORS_GAMMA[int(all_first_negative[0])])

        # Make figure 4 from the paper
        # calculate mean and std
        mean_n_negative = np.mean(all_n_negative, axis=0)
        std_n_negative = np.std(all_n_negative, axis=0)  
        mean_rel_diff = np.mean(all_rel_err, axis=0)
        std_rel_diff = np.std(all_rel_err, axis=0)  
        mean_noise_level = np.mean(all_mean_noise_level, axis=0)
        std_noise_level = np.std(all_mean_noise_level, axis=0)  

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

        # First subplot (left)
        ax1.errorbar(NOISE_FACTORS_GAMMA, [x[0] for x in mean_n_negative], yerr=[x[0] for x in std_n_negative], label=r'Orthogonal', color=colors[2], linestyle=':', fmt='o', markersize=2)
        ax1.errorbar(NOISE_FACTORS_GAMMA, [x[1] for x in mean_n_negative], yerr=[x[1] for x in std_n_negative], label=r'Positivity preserving', color=colors[5], linestyle='-.', fmt='o', markersize=2)
        ax1.errorbar(NOISE_FACTORS_GAMMA, [x[2] for x in mean_n_negative], yerr=[x[2] for x in std_n_negative], label=r'Orthogonal + backtracking', color=colors[1], linestyle='--', fmt='o', markersize=2)
        ax1.errorbar(NOISE_FACTORS_GAMMA, [x[3] for x in mean_n_negative], yerr=[x[3] for x in std_n_negative], label=r'Positivity preserving + backtracking', color=colors[4], linestyle=' ', fmt='o', markersize=2)

        ax1.set_ylabel(r'Fraction of negative concentrations / \%', fontsize=9)
        ax1.set_xlabel(r'Noise level $\gamma$', fontsize=9)
        ax1.set_ylim(-10, 110)
        ax1.set_xlim(-0.1, 1.1)
        ax1.set_title("Positivity", fontsize=9)

        # Second subplot (middle)
        ax2.errorbar(NOISE_FACTORS_GAMMA, [x[0] for x in mean_rel_diff], yerr=[x[0] for x in std_rel_diff], label=r'Orthogonal', color=colors[2], linestyle=':', fmt='o', markersize=2)
        ax2.errorbar(NOISE_FACTORS_GAMMA, [x[1] for x in mean_rel_diff], yerr=[x[1] for x in std_rel_diff], label=r'Positivity preserving', color=colors[5], linestyle='-.', fmt='o', markersize=2)
        ax2.errorbar(NOISE_FACTORS_GAMMA, [x[2] for x in mean_rel_diff], yerr=[x[2] for x in std_rel_diff], label=r'Orthogonal + backtracking', color=colors[1], linestyle='--', fmt='o', markersize=2)
        ax2.errorbar(NOISE_FACTORS_GAMMA, [x[3] for x in mean_rel_diff], yerr=[x[3] for x in std_rel_diff], label=r'Positivity preserving + backtracking', color=colors[4], linestyle=' ', fmt='o', markersize=2)
        ax2.errorbar(NOISE_FACTORS_GAMMA, [x[1] for x in mean_noise_level], yerr=[x[1] for x in std_noise_level], label=r'Noisy data error', color=colors[0], linestyle='-', fmt='o', markersize=2)
        ax2.set_yscale('log', base=10)
        ax2.set_ylabel(r'$\mathcal{L}_\mathrm{rel.}$ / \%', fontsize=9)
        ax2.set_xlabel(r'Noise level $\gamma$', fontsize=9)
        ax2.set_ylim(1e0, 1e6)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_title("Accuracy", fontsize=9)

        # generate data for backtracking in third plot
        cexp_corrected_backtrace = np.array([[0.15, 0.65]])
        C_pert_corr_naive = np.array([-0.075,  0.875])
        C0 = np.array([0.5, 0.3])
        dC_A_vals = np.linspace(-2, 2, 10)   
        dc_A_star_vals = -dC_A_vals
        C_vals_consistent = C0 + np.vstack([dC_A_vals, dc_A_star_vals]).T

        # Third subplot (right)
        ax3.scatter(C0[0], C0[1], marker='s', label=r'$c_\mathrm{0}$', s=20, c=colors[3],zorder=4)
        ax3.scatter(C_pert_corr_naive[0], C_pert_corr_naive[1], marker='v', label=r'$\tilde{c}_\mathrm{Proj.}$', s=20, c=colors[1],zorder=4)
        ax3.scatter(cexp_corrected_backtrace[0][0], cexp_corrected_backtrace[0][1], marker='o', label=r'$\bar{c}_\mathrm{Backtracking}$', s=20, c=colors[2],zorder=4)
        ax3.plot(C_vals_consistent[:, 0], C_vals_consistent[:, 1], color='k', linestyle='-', linewidth=0.5, label=r"Atom conserving $\Delta c$")
        ax3.set_xlim(-0.3, 1.3)
        ax3.set_ylim(-0.3, 1.3)
        ax3.set_xticks([0, 0.5, 1])
        ax3.set_yticks([0, 0.5, 1])
        ax3.plot([0.15, 0.15], [-0.5, 1.8], color='magenta', linestyle='--', linewidth=0.1)
        ax3.plot([-0.5, 1.8], [0.15, 0.15], color='magenta', linestyle='--', linewidth=0.1, label = 'Threshold')
        
        # Draw the arrow
        ax3.annotate('',  # No text
            xy=(cexp_corrected_backtrace[0][0], cexp_corrected_backtrace[0][1]),  
            xytext=(C_pert_corr_naive[0], C_pert_corr_naive[1]), 
            arrowprops=dict(
                arrowstyle=r'->',  
                lw=1.5,  
                color=colors[0]
            ),
            zorder=5
        )
        
        ax3.text(-0.225, 0.65, r'Track', color='k', fontsize=9)
        ax3.text(-0.2, 0.5, r'back', color='k', fontsize=9)
        ax3.set_xlabel(r'$c_A$', fontsize=9)
        ax3.set_ylabel(r'$c_{A^*}$', fontsize=9)
        ax3.set_title("Linear interpolation backtracking", fontsize=9)
        ax2.legend(bbox_to_anchor=(-0.25, -0.225), loc='upper center', ncol=2, frameon=True, edgecolor='black', fontsize=9)
        fig.subplots_adjust(wspace=0.4)

        ax1.set_aspect(0.01)
        ax2.set_aspect(1.2/6)
        ax3.set_aspect(1.0)
        ax1.text(-0.35, 1.05, "a)", transform=ax1.transAxes, size=12, weight='bold')
        ax2.text(-0.3, 1.05, "b)", transform=ax2.transAxes, size=12, weight='bold')
        ax3.text(-0.3, 1.05, "c)", transform=ax3.transAxes, size=12, weight='bold')

        print("Saving figure 4")
        plt.savefig('art/Figure4.png', dpi=300, bbox_inches='tight')

