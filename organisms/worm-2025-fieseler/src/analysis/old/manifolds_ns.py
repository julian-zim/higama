#region imports
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), '../../src/'))
from utils_worm import read_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from utils import plot_utils
from utils import train_utils
from utils import analysis_utils

import importlib

importlib.reload(read_utils)
importlib.reload(plot_utils)
importlib.reload(train_utils)
importlib.reload(analysis_utils)
#endregion


def main():
	#region setup
	organism = 'worm'

	ll = 'll00'
	data_variant = 'gcamp'
	model_variant = 'ns'
	manual = False
	full = True

	save = True

	file_ids = range(0, 33)
	# file_ids = [19, 20]
	bhl_selection = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']
	#endregion


	model_path = read_utils.get_model_path()
	for file_id in file_ids:
		neuron_labels = ['SMDDL', 'VB02', 'DB01', 'RMDDR', 'SMDDR', 'DD01']
		filename, traces, behavior, classes, aux = analysis_utils.read_data(organism, file_id)
		neuron_labels = [label for label in neuron_labels if label in traces.columns]
		metadata = (filename, model_path, model_variant, data_variant, manual, full)

		bom_params, hgm_params = analysis_utils.read_models(*metadata)
		((_, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss),
		 (_, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss)) = analysis_utils.read_predictions(*metadata)
		bom_manifolds, bom_manifold_vars_expl = analysis_utils.postprocess_manifold(bom_man, traces, bom_params)
		hgm_manifolds, hgm_manifold_vars_expl = analysis_utils.postprocess_manifold(bom_man, traces, bom_params)

		#region manifolds states
		(pca_modes, _, _,
		 pca_vars_expl) = train_utils.get_manifolds_pca(
			[traces],
			scale=False, n_pc=hgm_man[0].values.shape[1])
		(cca_modes, _, _, _, _, _,
		 cca_vars_expl, cca_vars_expl_rs) = train_utils.get_manifolds_cca(
			[traces], [classes[bhl_selection]],
			scale=False, n_cv=hgm_man[0].values.shape[1], n_pc=classes[bhl_selection].values.shape[1])
		(plsc_modes, _, _, _, _, _,
		 plsc_vars_expl) = train_utils.get_manifolds_plsc(
			[traces], [classes[bhl_selection]],
			scale=False, n_ls=hgm_man[0].values.shape[1])

		plot_utils.plot_manifolds([f'{str(file_id)}: {filename} s{s}' for s in range(len(bom_manifolds))],
								  [bom_manifolds[s] for s in range(len(bom_manifolds))],
								  [classes for s in range(len(bom_manifolds))],
								  bom_manifold_vars_expl, None,
								  title='bom manifold', plots_per_row=len(bom_manifolds), top_space=3.5,
								  save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_bom_states' if save else None)

		plot_utils.plot_manifolds([f'{str(file_id)}: {filename} s{s}' for s in range(len(hgm_manifolds))],
								  [hgm_manifolds[s] for s in range(len(hgm_manifolds))],
								  [classes for s in range(len(hgm_manifolds))],
								  hgm_manifold_vars_expl, None,
								  title='hgm manifold', plots_per_row=len(hgm_manifolds), top_space=3.5,
								  save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_hgm_states' if save else None)

		plot_utils.plot_manifolds([f'{str(file_id)}: {filename} s{s}' for s in range(len(pca_modes))],
								  [pca_modes[s] for s in range(len(pca_modes))],
								  [classes for s in range(len(pca_modes))],
								  pca_vars_expl, None,
								  title='pca manifold', plots_per_row=len(pca_modes), top_space=3.5,
								  save_path=f'../../out/plots/manifolds/{filename}/pca_states' if save else None)

		plot_utils.plot_manifolds([f'{str(file_id)}: {filename} s{s}' for s in range(len(cca_modes))],
								  [cca_modes[s] for s in range(len(cca_modes))],
								  [classes for s in range(len(cca_modes))],
								  cca_vars_expl, None,
								  title='cca manifold', plots_per_row=len(cca_modes), top_space=3.5,
								  save_path=f'../../out/plots/manifolds/{filename}/cca_states' if save else None)

		plot_utils.plot_manifolds([f'{str(file_id)}: {filename} s{s}' for s in range(len(plsc_modes))],
								  [plsc_modes[s] for s in range(len(plsc_modes))],
								  [classes for s in range(len(plsc_modes))],
								  plsc_vars_expl, None,
								  title='plsc manifold', plots_per_row=len(plsc_modes), top_space=3.5,
								  save_path=f'../../out/plots/manifolds/{filename}/plsc_states' if save else None)
		#endregion

		if len(neuron_labels) == 0: continue
		#region manifolds gating
		sigmoid = lambda x: 1 / (1 + np.exp(-x))
		neuron_ids = [traces.columns.get_loc(neuron) for neuron in neuron_labels]

		hgm_man_zs = [
			(hgm_man[s].values - hgm_man[s].values.mean(axis=0)) / np.where((N_std := hgm_man[s].values.std(axis=0)) == 0.0,
																			1.0, N_std) for s in range(len(hgm_man))]
		WMs = [hgm_params[s]['model']['WM'] for s in range(len(hgm_params))]
		gating_term = [pd.DataFrame(sigmoid(hgm_man_zs[s] @ WMs[s]), columns=traces.columns) for s in
					   range(len(hgm_man_zs))]

		plot_utils.plot_manifolds_gradient([f'{label}' for label in neuron_labels],
										   [bom_manifolds[0]] * len(neuron_ids),
										   [gating_term[0].iloc[:, n_id] for n_id in neuron_ids],
										   bom_manifold_vars_expl * len(neuron_ids), None,
										   title=f'{str(file_id)}: {filename} gcamp bom manifolds by gating of the different neurons',
										   plots_per_row=3, top_space=1,
										   save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_bom_gating' if save else None)

		plot_utils.plot_manifolds_gradient([f'{label}' for label in neuron_labels],
										   [hgm_manifolds[0]] * len(neuron_ids),
										   [gating_term[0].iloc[:, n_id] for n_id in neuron_ids],
										   hgm_manifold_vars_expl * len(neuron_ids), None,
										   title=f'{str(file_id)}: {filename} gcamp hgm manifolds by gating of the different neurons',
										   plots_per_row=3, top_space=1,
										   save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_hgm_gating' if save else None)

		plot_utils.plot_manifolds_gradient([f'{label}' for label in neuron_labels],
										   [pca_modes[0]] * len(neuron_ids),
										   [gating_term[0].iloc[:, n_id] for n_id in neuron_ids],
										   pca_vars_expl * len(neuron_ids), None,
										   title=f'{str(file_id)}: {filename} gcamp manifolds by gating of the different neurons',
										   plots_per_row=3, top_space=1,
										   save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_pca_gating' if save else None)

		plot_utils.plot_manifolds_gradient([f'{label}' for label in neuron_labels],
										   [cca_modes[0]] * len(neuron_ids),
										   [gating_term[0].iloc[:, n_id] for n_id in neuron_ids],
										   cca_vars_expl * len(neuron_ids), cca_vars_expl_rs * len(neuron_ids),
										   title=f'{str(file_id)}: {filename} gcamp manifolds by gating of the different neurons',
										   plots_per_row=3, top_space=1,
										   save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_cca_gating' if save else None)

		plot_utils.plot_manifolds_gradient([f'{label}' for label in neuron_labels],
										   [plsc_modes[0]] * len(neuron_ids),
										   [gating_term[0].iloc[:, n_id] for n_id in neuron_ids],
										   plsc_vars_expl * len(neuron_ids), None,
										   title=f'{str(file_id)}: {filename} gcamp manifolds by gating of the different neurons',
										   plots_per_row=3, top_space=1,
										   save_path=f'../../out/plots/manifolds/{filename}/{model_variant}_plsc_gating' if save else None)
	#endregion


if __name__ == '__main__':
	main()
