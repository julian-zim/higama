import sys
import os
import json
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA


sys.path.append(os.path.join(os.getcwd(), '../../src/'))
from utils_worm import read_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from utils import plot_utils
from utils import train_utils
from utils import analysis_utils
from models import model_utils


def main():
	file_ids = [19]
	# file_ids = list(range(0, 33+1))
	organism = 'worm'
	model_variant = 'ss'

	selection_bhl = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']

	for file_id in file_ids:
		filename, traces, eigenworms, discrete_states, (continuous_behaviors, kymograph) = analysis_utils.read_data(
			organism, file_id)

		# gating manifolds
		bom_params, hgm_params = analysis_utils.read_models(read_utils.get_model_path(), filename, model_variant,
															manual=False)
		((_, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss),
		 (_, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss)) = analysis_utils.read_predictions(
			read_utils.get_model_path(), filename, model_variant, manual=False)
		bom_manifolds, bom_manifold_vars_expl = analysis_utils.postprocess_manifold(bom_man, traces, bom_params)
		hgm_manifolds, hgm_manifold_vars_expl = analysis_utils.postprocess_manifold(hgm_man, traces, hgm_params)

		# pca
		pca_modes, _, _, pca_vars_expl = train_utils.get_manifolds_pca([traces],
																	   scale=False, n_pc=hgm_man[0].values.shape[1])
		# cca
		cca_modes, _, _, _, _, _, cca_vars_expl, cca_vars_expl_rs = train_utils.get_manifolds_cca(
			[traces], [discrete_states[selection_bhl]],
			scale=False, n_cv=hgm_man[0].values.shape[1], n_pc=discrete_states[selection_bhl].values.shape[1])
		# plsc
		plsc_modes, _, _, _, _, _, plsc_vars_expl = train_utils.get_manifolds_plsc(
			[traces], [discrete_states[selection_bhl]],
			scale=False, n_ls=hgm_man[0].values.shape[1])

		# residual modes
		max_modes = eigenworms.values.shape[1]
		pca_pres = PCA(n_components=hgm_pres[0].values.shape[1])
		pca_pres.fit(hgm_pres[0].values)

		cols = [f'Mode {i + 1}' for i in range(hgm_pres[0].shape[1])]

		# coloring
		sigmoid = lambda x: 1 / (1 + np.exp(-x))

		hgm_man_zs = (hgm_man[0].values - hgm_man[0].values.mean(axis=0)) / np.where(
			(N_std := hgm_man[0].values.std(axis=0)) == 0.0, 1.0, N_std)
		WM = hgm_params[0]['model']['WM']
		VB = hgm_params[0]['model']['VB']

		gating_space = [pd.DataFrame(sigmoid(hgm_man_zs @ WM) @ VB, columns=cols)]
		gating_space_modes = [pd.DataFrame(pca_pres.transform(gating_space[0].values), columns=cols)]

		# resss_ids = [hgm_resss[0].columns.get_loc(ss) for ss in hgm_resss[0].columns]

		# plot
		for modes, (var_expl, var_expl_rs), name in [
			(pca_modes, (pca_vars_expl, None), 'PCA'),
			(cca_modes, (cca_vars_expl, cca_vars_expl_rs), 'CCA'),
			(plsc_modes, (plsc_vars_expl, None), 'PLS-C'),
			(bom_manifolds, (bom_manifold_vars_expl, None), 'BOM'),
			(hgm_manifolds, (hgm_manifold_vars_expl, None), 'HGM')
		]:
			os.makedirs(f'manifolds_ss/{file_id}-{filename}', exist_ok=True)

			# plot by states
			plot_utils.plot_manifolds([''],
									  [modes[0]],
									  [discrete_states],
									  var_expl, var_expl_rs,
									  title=f'{file_id}-{filename}: {name} manifold',
									  plots_per_row=1, top_space=3.5,
									  save_path=f'manifolds_ss/{file_id}-{filename}/{file_id}-{filename}: {name} manifold')

			# plot by gradient
			plot_utils.plot_manifolds_gradient([f'Mode {ss + 1}' for ss in range(max_modes)],
											   [modes[0]] * max_modes,
											   [gating_space_modes[0].loc[:, [f'Mode {ss + 1}']] for ss in range(max_modes)],
											   var_expl * max_modes,
											   var_expl_rs * max_modes if var_expl_rs is not None else None,
											   title=f'{file_id}-{filename}: {name} manifold by gating',
											   plots_per_row=max_modes, top_space=1,
											   save_path=f'manifolds_ss/{file_id}-{filename}/{file_id}-{filename}: {name} manifold gating')





		# cols_plot = cols[:max_modes]
		# bom_pres_modes = [pd.DataFrame(pca_pres.transform(bom_pres[0].values), columns=cols)]
		# hgm_pres_modes = [pd.DataFrame(pca_pres.transform(hgm_pres[0].values), columns=cols)]
		#
		# bom_res_modes = [pd.DataFrame(pca_pres.transform(bom_res[0].values), columns=cols)]
		# hgm_res_modes = [pd.DataFrame(pca_pres.transform(hgm_res[0].values), columns=cols)]


if __name__ == '__main__':
	main()
