import os, sys
import numpy as np
import pandas as pd
import json
from utils_worm import read_utils
sys.path.append(os.path.join(os.getcwd(), '../../'))
from utils import train_utils
from models import model_utils


def read_data(file_id):
	path = os.path.join(read_utils.get_data_path(), 'exported_data_gcamp')

	# read
	(filename,
	 traces,
	 continuous_behaviors,
	 discrete_states, eigenworms,
	 kymograph) = read_utils.get_data(path, file_id)

	# clear nan values
	mask = eigenworms.notna().all(axis=1)

	traces = traces[mask]
	continuous_behaviors = continuous_behaviors[mask]
	eigenworms = eigenworms[mask]
	discrete_states = discrete_states[mask]
	kymograph = kymograph[mask]

	# naive ID unknown neurons
	traces.columns = pd.Index([
		f'{file_id:02d}_{n_id:03d}' if empty or type(label) is not str else label for n_id, (label, empty) in
		enumerate(zip(traces.columns, traces.columns == ''))
	])

	# preprocess
	pass

	return filename, traces, eigenworms, discrete_states, (continuous_behaviors, kymograph)


def read_models(filename, model_variant, manual):
	ll = 'll00'
	data_variant = 'gcamp'

	if manual:
		with open('../../hyperparams_manual.json', 'r') as f:
			reg_lambdas = json.load(f)
	else:
		with open('../../hyperparams.json', 'r') as f:
			reg_lambdas = json.load(f)['geometric'][ll]

	bom_folder = (f'{data_variant}_bom_{model_variant}_'
				  + '_'.join([f'{param.split('_')[0]}{reg_lambdas[model_variant]['bom'][param]:.2E}' for param in
							  reg_lambdas[model_variant]['bom'].keys()]))
	hgm_folder = (f'{data_variant}_hgm_{model_variant}_'
				  + '_'.join([f'{param.split('_')[0]}{reg_lambdas[model_variant]['hgm'][param]:.2E}' for param in
							  reg_lambdas[model_variant]['hgm'].keys()]))

	bom_params = train_utils.load_params(os.path.join(f'{read_utils.get_model_path()}/ALL', 'params', bom_folder),
										 filename)
	hgm_params = train_utils.load_params(os.path.join(f'{read_utils.get_model_path()}/ALL', 'params', hgm_folder),
										 filename)

	return bom_params, hgm_params


def read_predictions(filename, model_variant, manual):
	ll = 'll00'
	data_variant = 'gcamp'

	if manual:
		with open('../../hyperparams_manual.json', 'r') as f:
			reg_lambdas = json.load(f)
	else:
		with open('../../hyperparams.json', 'r') as f:
			reg_lambdas = json.load(f)['geometric'][ll]

	bom_folder = (f'{data_variant}_bom_{model_variant}_'
				  + '_'.join([f'{param.split('_')[0]}{reg_lambdas[model_variant]['bom'][param]:.2E}' for param in
							  reg_lambdas[model_variant]['bom'].keys()]))
	hgm_folder = (f'{data_variant}_hgm_{model_variant}_'
				  + '_'.join([f'{param.split('_')[0]}{reg_lambdas[model_variant]['hgm'][param]:.2E}' for param in
							  reg_lambdas[model_variant]['hgm'].keys()]))

	_, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss = train_utils.load_predictions(
		os.path.join(f'{read_utils.get_model_path()}/ALL', 'predictions', bom_folder), filename)
	_, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss = train_utils.load_predictions(
		os.path.join(f'{read_utils.get_model_path()}/ALL', 'predictions', hgm_folder), filename)

	return ((bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss),
			(hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss))


def postprocess_manifold(man, traces, params):
	manifolds = list()
	manifold_vars_expl = list()
	for s in range(len(params)):
		decode = lambda X: X @ model_utils.orthonormalize_qr(params[s]['model']['UN']).T
		manifold, var_explained = train_utils.postprocess_manifold(man[s].values, traces.values, decode,
																   scale=False, align=True)
		manifolds.append(pd.DataFrame(data=manifold, columns=man[s].columns))
		manifold_vars_expl.append(var_explained)

	return manifolds, manifold_vars_expl
