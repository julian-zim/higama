import sys
import os
import json
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA


sys.path.append(os.path.join(os.getcwd(), '../../src/'))
from utils_mouse import read_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from utils import plot_utils
from utils import train_utils
from utils import analysis_utils
from models import model_utils


def main():
	# file_ids = list(range(0, 12+1))
	file_ids = [2]
	organism = 'mouse'
	model_variant = 'ns'

	for file_id in file_ids:
		neuron_labels = ['02_8146', '02_2174', '02_7204', '02_233', '02_7412', '02_2214', '02_8097', '02_7331', '02_1408', '02_1246']
		filename, traces, behavior, classes, (continuous_behaviors, motion_masks, motion_pcs) = analysis_utils.read_data(organism, file_id)
		neuron_labels = [label for label in neuron_labels if label in traces.columns]

		# gating manifolds
		bom_params, hgm_params = analysis_utils.read_models(read_utils.get_model_path(), filename, model_variant,
															manual=False)
		((_, bom_man, bom_rec, bom_res, bom_pres, _, _),
		 (_, hgm_man, hgm_rec, hgm_res, hgm_pres, _, _)) = analysis_utils.read_predictions(read_utils.get_model_path(),
																						   filename, model_variant,
																						   manual=False)
		bom_manifolds, bom_manifold_vars_expl = analysis_utils.postprocess_manifold(bom_man, traces, bom_params)
		hgm_manifolds, hgm_manifold_vars_expl = analysis_utils.postprocess_manifold(hgm_man, traces, hgm_params)

		# pca
		pca_modes, _, _, pca_vars_expl = train_utils.get_manifolds_pca([traces],
																	   scale=False, n_pc=hgm_man[0].values.shape[1])

		# coloring
		sigmoid = lambda x: 1 / (1 + np.exp(-x))

		if len(neuron_labels) == 0: continue
		neuron_ids = [traces.columns.get_loc(neuron) for neuron in neuron_labels]

		sigmoid_terms = list()

		hgm_man_zs = (hgm_man[0].values - hgm_man[0].values.mean(axis=0)) / np.where(
			(N_std := hgm_man[0].values.std(axis=0)) == 0.0, 1.0, N_std)
		WM = hgm_params[0]['model']['WM']
		sigmoid_term = sigmoid(hgm_man_zs @ WM)
		sigmoid_terms.append(pd.DataFrame(data=sigmoid_term, columns=traces.columns))

		# plot
		for modes, (var_expl, var_expl_rs), name in [
			(pca_modes, (pca_vars_expl, None), 'PCA'),
			(bom_manifolds, (bom_manifold_vars_expl, None), 'BOM'),
			(hgm_manifolds, (hgm_manifold_vars_expl, None), 'HGM')
		]:
			os.makedirs(f'manifolds_ns/{file_id}-{filename}', exist_ok=True)

			# plot by states
			plot_utils.plot_manifolds_gradient([''],
									  [modes[0]],
									  [continuous_behaviors[['running_speed']]],
									  var_expl, var_expl_rs,
									  title=f'{file_id}-{filename}: {name} manifold by speed', plots_per_row=1, top_space=0.5,
									  save_path=f'manifolds_ns/{file_id}-{filename}/{file_id}-{filename}: {name} manifold')

			# plot by gradient
			plot_utils.plot_manifolds_gradient([f'{label}' for label in neuron_labels],
											   [modes[0]] * len(neuron_ids),
											   [sigmoid_terms[0].iloc[:, n_id] for n_id in neuron_ids],
											   var_expl * len(neuron_ids),
											   var_expl_rs * len(neuron_ids) if var_expl_rs is not None else None,
											   title=f'{file_id}-{filename}: {name} manifold by gating', plots_per_row=len(neuron_ids),
											   top_space=0.5,
											   save_path=f'manifolds_ns/{file_id}-{filename}/{file_id}-{filename}: {name} manifold by gating')


if __name__ == '__main__':
	main()
